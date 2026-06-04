"""
Dataset classes for training the Segmentation Quality Predictor (g_φ).

Provides:
- MQDataset: Mask Quality (MQ) Dataset with on-the-fly VQM generation.
- get_mq_dataloaders: Factory function for train/val/test dataloaders.
"""

import os
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from .vqm_generator import CorruptionConfig, VQMGenerator
from .weak_model_corruption import create_extended_vqm_generator


class MQDataset(Dataset):
    """
    Dataset for training the Segmentation Quality Predictor (g_φ).

    This dataset:
    1. Loads image-mask pairs from a CSV file.
    2. Generates corrupted masks on-the-fly using VQMGenerator.
    3. Returns (4-channel input, quality_score) pairs.

    The 4-channel input is: [RGB image, corrupted mask].
    """

    def __init__(
        self,
        csv_path: str,
        image_size: int = 224,
        corruption_config: Optional[CorruptionConfig] = None,
        spatial_transform: Optional[A.Compose] = None,
        image_transform: Optional[A.Compose] = None,
        is_train: bool = True,
        samples_per_image: int = 1,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Mask Quality (MQ) Dataset.

        Args:
            csv_path: Path to CSV with columns [image_path, mask_path].
            image_size: Target image size (square).
            corruption_config: Configuration for VQM generation.
            spatial_transform: Albumentations transforms applied to both image and mask.
            image_transform: Albumentations transforms applied only to image.
            is_train: If True, generate random corruptions. If False, use fixed corruptions.
            samples_per_image: Number of VQM samples to generate per image per epoch.
            seed: Random seed for reproducibility.
        """
        self.csv_path = csv_path
        self.image_size = image_size
        self.is_train = is_train
        self.samples_per_image = samples_per_image
        self.seed = seed

        # Load CSV.
        self.df = pd.read_csv(csv_path)

        # Validate CSV columns.
        required_cols = ["image_path", "mask_path"]
        # Also accept 'img_path' and 'seg_path' as alternatives.
        if "img_path" in self.df.columns:
            self.df = self.df.rename(columns={"img_path": "image_path"})
        if "seg_path" in self.df.columns:
            self.df = self.df.rename(columns={"seg_path": "mask_path"})

        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(
                    f"CSV must contain column '{col}'. Found: {self.df.columns.tolist()}"
                )

        # Set up transforms.
        self.spatial_transform = spatial_transform
        self.image_transform = (
            image_transform or self._get_default_image_transform(is_train)
        )

        # Load all GT masks for cross-image swap.
        print(
            f"Loading {len(self.df)} ground truth masks for cross-image swap..."
        )
        self.gt_masks = self._load_all_gt_masks()
        self.gt_mask_areas = np.array([m.sum() for m in self.gt_masks])

        # Initialize VQM generator.
        self.corruption_config = corruption_config or CorruptionConfig()
        self.vqm_generator = VQMGenerator(
            config=self.corruption_config,
            gt_masks=self.gt_masks,
            gt_mask_areas=self.gt_mask_areas,
        )

        # For reproducibility in validation.
        if seed is not None:
            self.base_rng = np.random.RandomState(seed)
        else:
            self.base_rng = np.random.RandomState()

    def _get_default_image_transform(self, is_train: bool) -> A.Compose:
        """
        Get default image-only transforms.

        Args:
            is_train: If True, apply data augmentation.

        Returns:
            Albumentations Compose object.
        """
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        if is_train:
            return A.Compose(
                [
                    A.RandomBrightnessContrast(p=0.5),
                    A.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.05,
                        p=0.3,
                    ),
                    A.GaussNoise(var_limit=(5, 25), p=0.2),
                    A.Normalize(mean=imagenet_mean, std=imagenet_std),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Normalize(mean=imagenet_mean, std=imagenet_std),
                    ToTensorV2(),
                ]
            )

    def _load_all_gt_masks(self) -> List[np.ndarray]:
        """
        Load all ground truth masks into memory for cross-image swap.

        Args:
            None

        Returns:
            List of ground truth masks.
        """
        masks = []
        for _, row in self.df.iterrows():
            mask_path = row["mask_path"]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                print(f"Warning: Could not load mask from {mask_path}")
                mask = np.zeros(
                    (self.image_size, self.image_size), dtype=np.uint8
                )
            else:
                mask = cv2.resize(mask, (self.image_size, self.image_size))
                _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

            masks.append(mask.astype(np.uint8))

        return masks

    def __len__(self) -> int:
        """
        Return total number of samples (images × samples_per_image).

        Args:
            None

        Returns:
            Total number of samples.
        """
        return len(self.df) * self.samples_per_image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training sample.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of:
                - input_tensor: 4-channel tensor [C, H, W] (RGB + mask)
                - quality_score: Scalar tensor with Dice coefficient
        """
        # Map idx to image index (accounting for samples_per_image)
        image_idx = idx // self.samples_per_image
        sample_idx = idx % self.samples_per_image

        row = self.df.iloc[image_idx]

        # Load image
        image = cv2.imread(row["image_path"])
        if image is None:
            raise ValueError(f"Could not load image from {row['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        # Get pre-loaded GT mask
        gt_mask = self.gt_masks[image_idx].copy()

        # Apply spatial transforms (to both image and mask)
        if self.spatial_transform:
            transformed = self.spatial_transform(image=image, mask=gt_mask)
            image = transformed["image"]
            gt_mask = transformed["mask"]

        # Generate corrupted mask with quality score.
        # # Use deterministic RNG for validation reproducibility.
        # if self.is_train:
        #     rng = np.random.RandomState()
        # else:
        #     # Deterministic based on idx for reproducible validation.
        #     rng = np.random.RandomState(self.seed + idx if self.seed else idx)

        # Earlier, we were using deterministic RNG for validation
        # reproducibility. But now, let's use stochastic RNG for both
        # training and validation to get more diverse corruptions.
        rng = np.random.RandomState()

        corrupted_mask, quality_score, corruption_name = self.vqm_generator(
            gt_mask=gt_mask,
            idx=image_idx,
            rng=rng,
        )

        # Apply image-only transforms.
        if self.image_transform:
            image = self.image_transform(image=image)["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Convert mask to tensor.
        mask_tensor = torch.from_numpy(corrupted_mask).unsqueeze(0).float()

        # Concatenate to create 4-channel input.
        input_tensor = torch.cat([image, mask_tensor], dim=0)

        # Quality score as tensor.
        quality_tensor = torch.tensor(quality_score, dtype=torch.float32)

        return input_tensor, quality_tensor

    def get_quality_distribution(self, n_samples: int = 1000) -> np.ndarray:
        """
        Sample the quality distribution for analysis.

        Args:
            n_samples: Number of samples to generate.

        Returns:
            Array of quality scores from n_samples corruptions.
        """
        qualities = []
        rng = np.random.RandomState(42)

        for i in range(n_samples):
            image_idx = i % len(self.df)
            gt_mask = self.gt_masks[image_idx]
            _, quality, _ = self.vqm_generator(gt_mask, image_idx, rng)
            qualities.append(quality)

        return np.array(qualities)


class MQDatasetWithWeakModels(Dataset):
    """
    Dataset for training the Segmentation Quality Predictor (g_φ).
    Extended version with weak model corruption support.

    This dataset:
    1. Loads image-mask pairs from a CSV file.
    2. Generates corrupted masks on-the-fly using VQMGenerator.
    3. Optionally uses weak model predictions as corruption source.
    4. Returns (4-channel input, quality_score) pairs.
    """

    def __init__(
        self,
        csv_path: str,
        image_size: int = 224,
        corruption_config=None,  # CorruptionConfig
        spatial_transform: Optional[A.Compose] = None,
        image_transform: Optional[A.Compose] = None,
        is_train: bool = True,
        samples_per_image: int = 1,
        seed: Optional[int] = None,
        # Weak model parameters
        weak_checkpoint_paths: Optional[List[str]] = None,
        weak_model_prob: float = 0.20,
        device: str = "cuda",
    ):
        """
        Initialize the Mask Quality (MQ) Dataset with optional weak model support.

        Args:
            csv_path: Path to CSV with columns [image_path, mask_path].
            image_size: Target image size (square).
            corruption_config: Configuration for VQM generation.
            spatial_transform: Albumentations transforms applied to both image and mask.
            image_transform: Albumentations transforms applied only to image.
            is_train: If True, generate random corruptions.
            samples_per_image: Number of VQM samples to generate per image per epoch.
            seed: Random seed for reproducibility.
            weak_checkpoint_paths: List of paths to weak model checkpoints.
            weak_model_prob: Probability of using weak model corruption.
            device: Device for weak model inference.
        """
        self.csv_path = csv_path
        self.image_size = image_size
        self.is_train = is_train
        self.samples_per_image = samples_per_image
        self.seed = seed
        self.device = device

        # Load CSV.
        self.df = pd.read_csv(csv_path)

        # Validate CSV columns.
        if "img_path" in self.df.columns:
            self.df = self.df.rename(columns={"img_path": "image_path"})
        if "seg_path" in self.df.columns:
            self.df = self.df.rename(columns={"seg_path": "mask_path"})

        for col in ["image_path", "mask_path"]:
            if col not in self.df.columns:
                raise ValueError(f"CSV must contain column '{col}'")

        # Set up transforms.
        self.spatial_transform = spatial_transform
        self.image_transform = (
            image_transform or self._get_default_image_transform(is_train)
        )

        # Load all GT masks for cross-image swap.
        print(
            f"Loading {len(self.df)} ground truth masks for cross-image swap..."
        )
        self.gt_masks = self._load_all_gt_masks()
        self.gt_mask_areas = np.array([m.sum() for m in self.gt_masks])

        # Initialize VQM generator (with optional weak model support).
        self._init_vqm_generator(
            corruption_config,
            weak_checkpoint_paths,
            weak_model_prob,
        )

        # For reproducibility.
        if seed is not None:
            self.base_rng = np.random.RandomState(seed)
        else:
            self.base_rng = np.random.RandomState()

    def _init_vqm_generator(
        self,
        corruption_config,
        weak_checkpoint_paths: Optional[List[str]],
        weak_model_prob: float,
    ):
        """
        Initialize the VQM generator with optional weak model support.

        Args:
            corruption_config: Configuration for VQM generation.
            weak_checkpoint_paths: List of paths to weak model checkpoints.
            weak_model_prob: Probability of using weak model corruption.
        """
        # Not needed because imported above already.
        # # Import here to avoid circular imports
        # # from vqm_generator import CorruptionConfig, VQMGenerator

        self.corruption_config = corruption_config or CorruptionConfig()

        if weak_checkpoint_paths:
            # Use extended generator with weak models.
            # from weak_model_corruption import create_extended_vqm_generator

            self.vqm_generator = create_extended_vqm_generator(
                corruption_config=self.corruption_config,
                gt_masks=self.gt_masks,
                weak_checkpoint_paths=weak_checkpoint_paths,
                weak_model_prob=weak_model_prob,
                device=self.device,
                image_size=self.image_size,
                use_cpu=False,
            )
            self.use_weak_models = True
            print(
                f"VQM generator initialized with weak model support "
                f"(prob={weak_model_prob})"
            )
        else:
            # Standard generator.
            self.vqm_generator = VQMGenerator(
                config=self.corruption_config,
                gt_masks=self.gt_masks,
                gt_mask_areas=self.gt_mask_areas,
            )
            self.use_weak_models = False

    def _get_default_image_transform(self, is_train: bool) -> A.Compose:
        """
        Get default image-only transforms.

        Args:
            is_train: Whether the dataset is for training.

        Returns:
            Albumentations transform.
        """
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        if is_train:
            return A.Compose(
                [
                    A.RandomBrightnessContrast(p=0.5),
                    A.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.05,
                        p=0.3,
                    ),
                    A.GaussNoise(var_limit=(5, 25), p=0.2),
                    A.Normalize(mean=imagenet_mean, std=imagenet_std),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Normalize(mean=imagenet_mean, std=imagenet_std),
                    ToTensorV2(),
                ]
            )

    def _load_all_gt_masks(self) -> List[np.ndarray]:
        """
        Load all ground truth masks into memory for cross-image swap.

        Args:
            None.

        Returns:
            List of ground truth masks.
        """
        masks = []
        for _, row in self.df.iterrows():
            mask_path = row["mask_path"]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                print(f"Warning: Could not load mask from {mask_path}")
                mask = np.zeros(
                    (self.image_size, self.image_size), dtype=np.uint8
                )
            else:
                mask = cv2.resize(mask, (self.image_size, self.image_size))
                _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

            masks.append(mask.astype(np.uint8))

        return masks

    def __len__(self) -> int:
        """
        Return total number of samples (images × samples_per_image).

        Returns:
            Integer representing total number of samples.
        """
        return len(self.df) * self.samples_per_image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training sample.


        Returns:
            Tuple of:
                - input_tensor: 4-channel tensor [C, H, W] (RGB + mask).
                - quality_score: Scalar tensor with Dice coefficient.
        """
        # Map idx to image index.
        image_idx = idx // self.samples_per_image

        row = self.df.iloc[image_idx]

        # Load image.
        image = cv2.imread(row["image_path"])
        if image is None:
            raise ValueError(f"Could not load image from {row['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        # Get pre-loaded GT mask.
        gt_mask = self.gt_masks[image_idx].copy()

        # Apply spatial transforms (to both image and mask).
        if self.spatial_transform:
            transformed = self.spatial_transform(image=image, mask=gt_mask)
            image = transformed["image"]
            gt_mask = transformed["mask"]

        # Store raw image for weak model (before normalization).
        image_for_weak = image.copy()

        # Generate corrupted mask with quality score.
        rng = np.random.RandomState()

        if self.use_weak_models:
            # Pass image for potential weak model corruption.
            corrupted_mask, quality_score, corruption_name = (
                self.vqm_generator(
                    gt_mask=gt_mask,
                    idx=image_idx,
                    rng=rng,
                    image=image_for_weak,
                )
            )
        else:
            corrupted_mask, quality_score, corruption_name = (
                self.vqm_generator(
                    gt_mask=gt_mask,
                    idx=image_idx,
                    rng=rng,
                )
            )

        # Apply image-only transforms.
        if self.image_transform:
            image = self.image_transform(image=image)["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Convert mask to tensor.
        mask_tensor = torch.from_numpy(corrupted_mask).unsqueeze(0).float()

        # Concatenate to create 4-channel input.
        input_tensor = torch.cat([image, mask_tensor], dim=0)

        # Quality score as tensor.
        quality_tensor = torch.tensor(quality_score, dtype=torch.float32)

        return input_tensor, quality_tensor


def get_spatial_transforms(image_size: int, is_train: bool) -> A.Compose:
    """
    Get spatial transforms that apply to both image and mask.

    Args:
        image_size: Target image size.
        is_train: Whether the dataset is for training.

    Returns:
        Albumentations transform.
    """
    if is_train:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
            ]
        )


def get_mq_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: Optional[str] = None,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    corruption_config: Optional[CorruptionConfig] = None,
    samples_per_image_train: int = 1,
    samples_per_image_val: int = 1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create dataloaders for g_φ training.

    Args:
        train_csv: Path to training CSV.
        val_csv: Path to validation CSV.
        test_csv: Optional path to test CSV.
        image_size: Target image size.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for faster GPU transfer.
        corruption_config: VQM generation configuration.
        samples_per_image_train: VQM samples per image per epoch (train).
        samples_per_image_val: VQM samples per image per epoch (val).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Create datasets
    train_dataset = MQDataset(
        csv_path=train_csv,
        image_size=image_size,
        corruption_config=corruption_config,
        spatial_transform=get_spatial_transforms(image_size, is_train=True),
        is_train=True,
        samples_per_image=samples_per_image_train,
        seed=None,  # Random for training.
    )

    val_dataset = MQDataset(
        csv_path=val_csv,
        image_size=image_size,
        corruption_config=corruption_config,
        spatial_transform=get_spatial_transforms(image_size, is_train=False),
        is_train=False,
        samples_per_image=samples_per_image_val,
        seed=seed,  # Deterministic for validation.
    )

    test_dataset = None
    if test_csv is not None:
        test_dataset = MQDataset(
            csv_path=test_csv,
            image_size=image_size,
            corruption_config=corruption_config,
            spatial_transform=get_spatial_transforms(
                image_size, is_train=False
            ),
            is_train=False,
            samples_per_image=samples_per_image_val,
            seed=seed,
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # For consistent batch sizes.
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    # Print dataset info.
    print("Dataset sizes:")
    print(
        f"  Train: {len(train_dataset)} samples ({len(train_dataset.df)} images × {samples_per_image_train})"
    )
    print(
        f"  Val:   {len(val_dataset)} samples ({len(val_dataset.df)} images × {samples_per_image_val})"
    )
    if test_dataset:
        print(
            f"  Test:  {len(test_dataset)} samples ({len(test_dataset.df)} images × {samples_per_image_val})"
        )

    return train_loader, val_loader, test_loader


def get_mq_dataloaders_with_weak_models(
    train_csv: str,
    val_csv: str,
    test_csv: Optional[str] = None,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    corruption_config=None,
    samples_per_image_train: int = 1,
    samples_per_image_val: int = 1,
    seed: int = 42,
    # Weak model parameters.
    weak_checkpoint_paths: Optional[List[str]] = None,
    weak_model_prob: float = 0.20,
    device: str = "cuda",
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create dataloaders for g_φ training with optional weak model support.
    """
    # Not needed because imported above already.
    # from vqm_generator import CorruptionConfig

    def get_spatial_transforms(image_size: int, is_train: bool) -> A.Compose:
        if is_train:
            return A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                ]
            )
        else:
            return A.Compose([A.Resize(image_size, image_size)])

    # Create datasets.
    train_dataset = MQDatasetWithWeakModels(
        csv_path=train_csv,
        image_size=image_size,
        corruption_config=corruption_config,
        spatial_transform=get_spatial_transforms(image_size, is_train=True),
        is_train=True,
        samples_per_image=samples_per_image_train,
        seed=None,
        weak_checkpoint_paths=weak_checkpoint_paths,
        weak_model_prob=weak_model_prob,
        device=device,
    )

    # Validation doesn't use weak models (for consistency).
    val_dataset = MQDatasetWithWeakModels(
        csv_path=val_csv,
        image_size=image_size,
        corruption_config=corruption_config,
        spatial_transform=get_spatial_transforms(image_size, is_train=False),
        is_train=False,
        samples_per_image=samples_per_image_val,
        seed=seed,
        weak_checkpoint_paths=None,  # No weak models for validation.
        weak_model_prob=0.0,
        device=device,
    )

    test_dataset = None
    if test_csv is not None:
        test_dataset = MQDatasetWithWeakModels(
            csv_path=test_csv,
            image_size=image_size,
            corruption_config=corruption_config,
            spatial_transform=get_spatial_transforms(
                image_size, is_train=False
            ),
            is_train=False,
            samples_per_image=samples_per_image_val,
            seed=seed,
            weak_checkpoint_paths=None,  # No weak models for testing.
            weak_model_prob=0.0,
            device=device,
        )

    # Create dataloaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    # Print dataset info.
    print("Dataset sizes:")
    print(
        f"  Train: {len(train_dataset)} samples "
        f"({len(train_dataset.df)} images × {samples_per_image_train})"
    )
    if train_dataset.use_weak_models:
        print(
            f"  Train using weak model corruption with prob={weak_model_prob}"
        )
    print(f"  Val:   {len(val_dataset)} samples")
    if test_dataset:
        print(f"  Test:  {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader


# =============================================================================
# Testing
# =============================================================================


def test_mq_dataset():
    """
    Test the Mask Quality (MQ) dataset with a dummy CSV.
    """
    import tempfile

    import matplotlib.pyplot as plt

    # Create dummy data.
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy images and masks.
        n_samples = 10
        image_paths = []
        mask_paths = []

        for i in range(n_samples):
            # Create dummy image.
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img_path = os.path.join(tmpdir, f"img_{i}.png")
            cv2.imwrite(img_path, img)
            image_paths.append(img_path)

            # Create dummy mask (circle).
            mask = np.zeros((256, 256), dtype=np.uint8)
            cv2.circle(mask, (128 + i * 5, 128), 50, 255, -1)
            mask_path = os.path.join(tmpdir, f"mask_{i}.png")
            cv2.imwrite(mask_path, mask)
            mask_paths.append(mask_path)

        # Create CSV.
        csv_path = os.path.join(tmpdir, "data.csv")
        df = pd.DataFrame(
            {
                "image_path": image_paths,
                "mask_path": mask_paths,
            }
        )
        df.to_csv(csv_path, index=False)

        # Create dataset.
        dataset = MQDataset(
            csv_path=csv_path,
            image_size=224,
            is_train=True,
            samples_per_image=3,
        )

        print(f"Dataset length: {len(dataset)}")

        # Get some samples.
        qualities = []
        for i in range(min(30, len(dataset))):
            input_tensor, quality = dataset[i]
            qualities.append(quality.item())

            if i < 6:
                print(
                    f"Sample {i}: quality={quality.item():.3f}, shape={input_tensor.shape}"
                )

        print(f"\nQuality stats from {len(qualities)} samples:")
        print(f"  Mean: {np.mean(qualities):.3f}")
        print(f"  Std:  {np.std(qualities):.3f}")
        print(f"  Min:  {np.min(qualities):.3f}")
        print(f"  Max:  {np.max(qualities):.3f}")

        # Test quality distribution.
        print("\nSampling quality distribution (1000 samples)...")
        dist = dataset.get_quality_distribution(n_samples=1000)
        plt.figure(figsize=(8, 4))
        plt.hist(dist, bins=20, edgecolor="black")
        plt.xlabel("Quality Score")
        plt.ylabel("Count")
        plt.title("Quality Distribution")
        plt.savefig("mq_dataset_quality_dist.png", dpi=100)
        print("Saved to mq_dataset_quality_dist.png")


if __name__ == "__main__":
    test_mq_dataset()
