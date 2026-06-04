"""
Weak Model Corruption for Variable Quality Mask Generation.

This module extends the VQM generator to use predictions from partially-trained
segmentation models as a corruption source, addressing the train-test distribution
mismatch between synthetic corruptions and real neural network errors.

Usage:
    1. First, train weak models using train_weak_models().
    2. Then, pass the checkpoint paths to VQMGenerator via WeakModelCorruptor.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# segmentation_models_pytorch for U-Net.
try:
    import segmentation_models_pytorch as smp

    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print(
        "Warning: segmentation_models_pytorch not available. "
        "Install with: pip install segmentation-models-pytorch"
    )


def dice_coefficient(
    pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5
) -> float:
    """
    Compute Dice coefficient between two binary masks.

    Args:
        pred: Predicted/corrupted mask.
        target: Ground truth mask.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Dice coefficient in [0, 1].
    """
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()
    intersection = np.logical_and(pred, target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


class WeakModelCorruptor:
    """
    Generates corrupted masks using predictions from weak (partially-trained) models.

    This creates more realistic "neural network error" patterns compared to
    morphological corruptions, helping g_φ generalize better to real
    pseudo-labels.

    By default, inference runs on CPU to allow compatibility with multi-worker
    DataLoaders. Set use_cpu=False and num_workers=0 for GPU inference.
    """

    def __init__(
        self,
        checkpoint_paths: List[str],
        encoder_name: str = "resnet18",
        in_channels: int = 3,
        device: str = "cuda",
        image_size: int = 224,
        use_cpu: bool = True,  # Default to CPU for DataLoader compatibility.
    ):
        """
        Initialize the weak model corruptor.

        Args:
            checkpoint_paths: List of paths to weak model checkpoints.
            encoder_name: Encoder architecture (default: resnet18 for speed).
            in_channels: Number of input channels.
            device: Device for loading checkpoints (models moved to inference device).
            image_size: Expected input image size.
            use_cpu: If True, run inference on CPU (required for num_workers > 0).
        """
        if not SMP_AVAILABLE:
            raise ImportError(
                "segmentation_models_pytorch required for WeakModelCorruptor"
            )

        self.device = device
        self.use_cpu = use_cpu
        self.inference_device = "cpu" if use_cpu else device
        self.image_size = image_size
        self.models: List[nn.Module] = []
        self.checkpoint_info: List[Dict] = []

        # Load each checkpoint.
        for ckpt_path in checkpoint_paths:
            ckpt_path = Path(ckpt_path)
            if not ckpt_path.exists():
                print(f"Warning: Checkpoint not found: {ckpt_path}")
                continue

            # Create model (without pretrained weights - matches training).
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,  # No pretrained weights.
                in_channels=in_channels,
                classes=1,
            )

            # Load checkpoint.
            checkpoint = torch.load(
                ckpt_path, map_location="cpu", weights_only=False
            )

            # Handle different checkpoint formats.
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                    epoch = checkpoint.get("epoch", "unknown")
                    dice = checkpoint.get(
                        "val_dice", checkpoint.get("train_dice", "unknown")
                    )
                elif "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                    epoch = checkpoint.get("epoch", "unknown")
                    dice = "unknown"
                else:
                    # Assume it's just the state dict.
                    model.load_state_dict(checkpoint)
                    epoch = "unknown"
                    dice = "unknown"
            else:
                model.load_state_dict(checkpoint)
                epoch = "unknown"
                dice = "unknown"

            model.to(self.inference_device)
            model.eval()
            self.models.append(model)
            self.checkpoint_info.append(
                {
                    "path": str(ckpt_path),
                    "epoch": epoch,
                    "dice": dice,
                }
            )
            print(f"Loaded weak model: epoch={epoch}, dice={dice}")

        if not self.models:
            raise ValueError("No valid checkpoints loaded!")

        device_str = "CPU" if self.use_cpu else self.device
        print(
            f"WeakModelCorruptor initialized with {len(self.models)} models "
            f"(inference on {device_str})"
        )

    @torch.no_grad()
    def __call__(
        self,
        image: np.ndarray,
        gt_mask: np.ndarray,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, float, str]:
        """
        Generate a corrupted mask using a weak model prediction.

        Args:
            image: Input image (H, W, C) in [0, 255] uint8 or (H, W, C) float.
            gt_mask: Ground truth mask (H, W) binary.
            rng: Random state for model selection.

        Returns:
            Tuple of (corrupted_mask, quality_score, corruption_name).
        """
        # Select a random weak model.
        model_idx = rng.randint(0, len(self.models))
        model = self.models[model_idx]

        # Preprocess image.
        if image.dtype == np.uint8:
            image_float = image.astype(np.float32) / 255.0
        else:
            image_float = image.astype(np.float32)

        # Resize if needed.
        if image_float.shape[:2] != (self.image_size, self.image_size):
            image_float = cv2.resize(
                image_float, (self.image_size, self.image_size)
            )

        # Normalize (ImageNet stats, though model wasn't pretrained - keeps consistency).
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_norm = (image_float - mean) / std

        # To tensor: (H, W, C) -> (1, C, H, W).
        image_tensor = (
            torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0).float()
        )
        image_tensor = image_tensor.to(self.inference_device)

        # Get prediction.
        logits = model(image_tensor)
        pred = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Binarize.
        corrupted_mask = (pred > 0.5).astype(np.uint8)

        # Resize mask back if needed.
        if corrupted_mask.shape != gt_mask.shape:
            corrupted_mask = cv2.resize(
                corrupted_mask,
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Compute actual Dice against ground truth.
        quality_score = dice_coefficient(corrupted_mask, gt_mask)

        # Corruption name includes model info.
        info = self.checkpoint_info[model_idx]
        corruption_name = f"weak_model_epoch_{info['epoch']}"

        return corrupted_mask, quality_score, corruption_name

    def get_quality_range(self) -> Tuple[float, float]:
        """
        Return expected quality range based on loaded models.
        """
        dices = [
            info["dice"]
            for info in self.checkpoint_info
            if info["dice"] != "unknown"
        ]
        if dices:
            return min(dices), max(dices)

        return 0.0, 1.0


# =============================================================================
# Training script for weak models.
# =============================================================================


class SimpleSegDataset(Dataset):
    """
    Simple dataset for training weak models.
    """

    def __init__(
        self,
        csv_path: str,
        image_size: int = 224,
        augment: bool = True,
    ):
        import albumentations as A
        import pandas as pd
        from albumentations.pytorch import ToTensorV2

        self.df = pd.read_csv(csv_path)
        self.image_size = image_size

        # Handle column name variations.
        if "img_path" in self.df.columns:
            self.df = self.df.rename(columns={"img_path": "image_path"})
        if "seg_path" in self.df.columns:
            self.df = self.df.rename(columns={"seg_path": "mask_path"})

        # Transforms.
        if augment:
            self.transform = A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=20, p=0.5),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = cv2.imread(row["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"].unsqueeze(0).float()

        return image, mask


def train_weak_models(
    train_csv: str,
    output_dir: str,
    save_epochs: List[int] = [1, 3, 5, 7],
    max_epochs: int = 10,
    encoder_name: str = "resnet18",
    batch_size: int = 16,
    lr: float = 1e-3,
    image_size: int = 224,
    device: str = "cuda",
    seed: int = 42,
) -> List[str]:
    """
    Train a weak segmentation model and save checkpoints at specified epochs.

    The model is trained WITHOUT pretrained weights to ensure slow convergence
    and create diverse quality levels across epochs.

    Args:
        train_csv: Path to training CSV.
        output_dir: Directory to save checkpoints.
        save_epochs: List of epochs at which to save checkpoints.
        max_epochs: Maximum epochs to train.
        encoder_name: Encoder architecture.
        batch_size: Training batch size.
        lr: Learning rate.
        image_size: Input image size.
        device: Training device.
        seed: Random seed.

    Returns:
        List of paths to saved checkpoints.
    """
    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch required")

    import torch.optim as optim

    # Set seed.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create output directory.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset and dataloader.
    dataset = SimpleSegDataset(train_csv, image_size=image_size, augment=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Create model **without** pretrained weights.
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,  # Random initialization!
        in_channels=3,
        classes=1,
    )
    model = model.to(device)

    # Loss and optimizer.
    criterion = smp.losses.DiceLoss(mode="binary")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    saved_paths = []

    print(
        f"Training weak model (NO pretrained weights) for {max_epochs} epochs..."
    )
    print(f"Will save at epochs: {save_epochs}")

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{max_epochs}")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            # Compute Dice for monitoring.
            with torch.no_grad():
                pred = (torch.sigmoid(outputs) > 0.5).float()
                intersection = (pred * masks).sum()
                dice = (2 * intersection + 1e-5) / (
                    pred.sum() + masks.sum() + 1e-5
                )

            epoch_loss += loss.item()
            epoch_dice += dice.item()
            n_batches += 1

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "dice": f"{dice.item():.4f}"}
            )

        avg_loss = epoch_loss / n_batches
        avg_dice = epoch_dice / n_batches
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Dice={avg_dice:.4f}")

        # Save checkpoint if this is a target epoch.
        if epoch in save_epochs:
            ckpt_path = output_dir / f"weak_model_epoch_{epoch}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "train_dice": avg_dice,
                    "train_loss": avg_loss,
                    "encoder_name": encoder_name,
                },
                ckpt_path,
            )
            saved_paths.append(str(ckpt_path))
            print(f"  -> Saved checkpoint: {ckpt_path}")

    print(
        f"\nWeak model training complete. Saved {len(saved_paths)} checkpoints."
    )
    return saved_paths


# =============================================================================
# Integration with VQMGenerator.
# =============================================================================


def create_extended_vqm_generator(
    corruption_config,
    gt_masks: List[np.ndarray],
    weak_checkpoint_paths: Optional[List[str]] = None,
    weak_model_prob: float = 0.20,
    device: str = "cuda",
    image_size: int = 224,
    use_cpu: bool = True,
):
    """
    Create a VQMGenerator with optional weak model corruption support.

    This is a factory function that creates the standard VQMGenerator and
    optionally wraps it to include weak model predictions.

    Args:
        corruption_config: CorruptionConfig for standard corruptions.
        gt_masks: List of ground truth masks for cross-image swap.
        weak_checkpoint_paths: Optional list of weak model checkpoint paths.
        weak_model_prob: Probability of using weak model (vs standard corruption).
        device: Device for loading checkpoints.
        image_size: Image size for weak model.
        use_cpu: If True, run inference on CPU for DataLoader compatibility.

    Returns:
        Extended VQMGenerator instance.
    """
    from .vqm_generator import VQMGenerator

    # Create base generator.
    base_generator = VQMGenerator(
        config=corruption_config,
        gt_masks=gt_masks,
        gt_mask_areas=np.array([m.sum() for m in gt_masks]),
    )

    # If no weak models, return base generator.
    if not weak_checkpoint_paths:
        return base_generator

    # Create weak model corruptor.
    weak_corruptor = WeakModelCorruptor(
        checkpoint_paths=weak_checkpoint_paths,
        device=device,
        image_size=image_size,
        use_cpu=use_cpu,
    )

    # Create wrapper class.
    class ExtendedVQMGenerator:
        """
        VQMGenerator extended with weak model corruption.
        """

        def __init__(self, base, weak, weak_prob):
            self.base = base
            self.weak = weak
            self.weak_prob = weak_prob

        def __call__(
            self, gt_mask, idx, rng=None, image=None, force_category=None
        ):
            """
            Generate corrupted mask.

            Args:
                gt_mask: Ground truth mask
                idx: Sample index
                rng: Random state
                image: Input image (required for weak model corruption)
                force_category: Force specific category (bypasses weak model)
            """
            if rng is None:
                rng = np.random.RandomState()

            # Decide whether to use weak model.
            use_weak = (
                image is not None
                and force_category is None
                and rng.random() < self.weak_prob
            )

            if use_weak:
                return self.weak(image, gt_mask, rng)
            else:
                return self.base(gt_mask, idx, rng, force_category)

        def set_gt_masks(self, gt_masks):
            self.base.set_gt_masks(gt_masks)

    return ExtendedVQMGenerator(
        base_generator, weak_corruptor, weak_model_prob
    )


# =============================================================================
# CLI for training weak models.
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train weak models for VQM generation"
    )
    parser.add_argument(
        "--train_csv", type=str, required=True, help="Training CSV path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./weak_checkpoints",
        help="Output directory",
    )
    parser.add_argument(
        "--save_epochs",
        type=int,
        nargs="+",
        default=[1, 3, 5, 7],
        help="Epochs to save",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="Max training epochs"
    )
    parser.add_argument(
        "--encoder_name",
        type=str,
        default="resnet18",
        help="Encoder architecture",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--image_size", type=int, default=224, help="Image size"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    saved = train_weak_models(
        train_csv=args.train_csv,
        output_dir=args.output_dir,
        save_epochs=args.save_epochs,
        max_epochs=args.max_epochs,
        encoder_name=args.encoder_name,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        device=args.device,
        seed=args.seed,
    )

    print("\nTo use these checkpoints, add to your g_φ training:")
    print(f"  --weak_checkpoints {' '.join(saved)}")
