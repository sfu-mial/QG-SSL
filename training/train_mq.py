"""
Training script for the Segmentation Quality Predictor (g_φ).

This script trains g_φ on synthetically generated Variable Quality Masks (VQMs)
to predict the quality (Dice coefficient) of segmentation masks.
"""

import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

# Optional: experiment tracking.
try:
    from comet_ml import Experiment

    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

# Local imports.
import sys

sys.path.append(str(Path(__file__).parent.parent))

from data.mq_dataset import get_mq_dataloaders
from data.vqm_generator import CorruptionConfig
from models.quality_predictor import create_quality_predictor


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train Segmentation Quality Predictor (g_φ)"
    )

    # Data arguments.
    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="Path to training CSV with image_path, mask_path columns",
    )
    parser.add_argument(
        "--val_csv", type=str, required=True, help="Path to validation CSV"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help="Path to test CSV (optional)",
    )

    # Model arguments.
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18.a1_in1k",
        help="Backbone model name from timm",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained weights",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.15, help="Dropout rate"
    )

    # Training arguments.
    parser.add_argument(
        "--epochs", type=int, default=150, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="Weight decay"
    )
    parser.add_argument(
        "--image_size", type=int, default=224, help="Input image size"
    )
    parser.add_argument(
        "--samples_per_image",
        type=int,
        default=50,
        help="VQM samples per image per epoch",
    )
    parser.add_argument(
        "--samples_per_image_val",
        type=int,
        default=5,
        help="VQM samples per image per epoch for validation",
    )

    parser.add_argument(
        "--loss_func",
        type=str,
        default="smoothl1",
        choices=["smoothl1", "mse"],
        help="Loss function (smoothl1 or mse)",
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "onecycle"],
        help="Scheduler (cosine or onecycle)",
    )

    # Corruption config.
    parser.add_argument(
        "--identity_prob",
        type=float,
        default=0.20,
        help="Probability of identity (no corruption)",
    )
    parser.add_argument(
        "--weak_checkpoint_paths",
        type=str,
        nargs="*",
        default=None,
        help="Paths to weak model checkpoints for corruption diversity",
    )
    parser.add_argument(
        "--weak_model_prob",
        type=float,
        default=0.05,
        help="Probability of using weak model predictions as corruption",
    )
    parser.add_argument(
        "--cross_image_prob",
        type=float,
        default=0.10,
        help="Probability of cross-image swap",
    )

    # System arguments.
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    # parser.add_argument(
    #     "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    # )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output arguments.
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/mq",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name for logging",
    )

    # Logging arguments.
    parser.add_argument(
        "--use_comet", action="store_true", help="Use Comet.ml for logging"
    )
    parser.add_argument(
        "--comet_project",
        type=str,
        default="seg-quality-predictor",
        help="Comet.ml project name",
    )

    # Early stopping.
    parser.add_argument(
        "--patience", type=int, default=25, help="Early stopping patience"
    )

    return parser.parse_args()


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class EarlyStopper:
    """
    Early stopping handler.

    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change in loss to consider an improvement.
    """

    def __init__(self, patience: int, min_delta: float = 1e-4):
        """
        Initialize the early stopper.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def should_stop(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Validation loss.

        Returns:
            True if training should stop, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epoch: int,
    accelerator: Accelerator,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: Model to train.
        dataloader: DataLoader for training.
        optimizer: Optimizer.
        criterion: Loss function.
        epoch: Current epoch.
        accelerator: Accelerator object.

    Returns:
        Tuple of (average loss, average MAE).
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Train]")

    for inputs, targets in pbar:
        # inputs = inputs.to(device)
        # targets = targets.to(device).unsqueeze(1)
        targets = targets.unsqueeze(1)

        # Forward pass.
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass.
        optimizer.zero_grad()
        # loss.backward()
        with accelerator.autocast():
            accelerator.backward(loss)

        # Gradient clipping.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Metrics.
        with torch.no_grad():
            mae = torch.abs(outputs - targets).mean()

        total_loss += loss.item()
        total_mae += mae.item()
        n_batches += 1

        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "mae": f"{mae.item():.4f}"}
        )

    return total_loss / n_batches, total_mae / n_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    epoch: int,
) -> Tuple[float, float, float]:
    """
    Validate the model.

    Args:
        model: Model to validate.
        dataloader: DataLoader for validation.
        criterion: Loss function.
        epoch: Current epoch.

    Returns:
        Tuple of (average loss, average MAE, correlation).
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n_batches = 0

    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Val]")

    for inputs, targets in pbar:
        # inputs = inputs.to(device)
        # targets = targets.to(device).unsqueeze(1)
        targets = targets.unsqueeze(1)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        mae = torch.abs(outputs - targets).mean()

        total_loss += loss.item()
        total_mae += mae.item()
        n_batches += 1

        all_preds.extend(outputs.cpu().numpy().flatten())
        all_targets.extend(targets.cpu().numpy().flatten())

        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "mae": f"{mae.item():.4f}"}
        )

    # Compute correlation.
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    if all_preds.std() > 1e-6 and all_targets.std() > 1e-6:
        correlation = np.corrcoef(all_preds, all_targets)[0, 1]
    else:
        correlation = 0.0

    return total_loss / n_batches, total_mae / n_batches, correlation


@torch.no_grad()
def test_image_dependence(
    model: nn.Module,
    dataloader: DataLoader,
    n_batches: int = 5,
):
    """
    Test if model actually uses image content or just looks at mask.

    Compares predictions on normal inputs vs. blacked-out images.
    If outputs are similar, model is ignoring the image (bad!).

    Args:
        model: Model to test.
        dataloader: DataLoader for testing.
        n_batches: Number of batches to use for testing.

    Returns:
        Dictionary containing test results.
    """
    model.eval()

    normal_outputs = []
    blackout_outputs = []
    mask_only_outputs = []

    for i, (inputs, targets) in enumerate(dataloader):
        if i >= n_batches:
            break

        # inputs = inputs.to(device)

        # Normal prediction.
        out_normal = model(inputs)
        normal_outputs.extend(out_normal.cpu().numpy().flatten())

        # Blackout image (zero RGB, keep mask).
        inputs_blackout = inputs.clone()
        inputs_blackout[:, :3, :, :] = 0
        out_blackout = model(inputs_blackout)
        blackout_outputs.extend(out_blackout.cpu().numpy().flatten())

        # Random image (shuffle RGB across batch, keep mask).
        inputs_shuffled = inputs.clone()
        perm = torch.randperm(inputs.size(0))
        inputs_shuffled[:, :3, :, :] = inputs[perm, :3, :, :]
        out_shuffled = model(inputs_shuffled)
        mask_only_outputs.extend(out_shuffled.cpu().numpy().flatten())

    normal_outputs = np.array(normal_outputs)
    blackout_outputs = np.array(blackout_outputs)
    mask_only_outputs = np.array(mask_only_outputs)

    # Compute differences.
    blackout_diff = np.abs(normal_outputs - blackout_outputs).mean()
    shuffled_diff = np.abs(normal_outputs - mask_only_outputs).mean()

    # Correlation between normal and blackout (should be LOW if using image).
    blackout_corr = np.corrcoef(normal_outputs, blackout_outputs)[0, 1]
    shuffled_corr = np.corrcoef(normal_outputs, mask_only_outputs)[0, 1]

    print("\n" + "=" * 50)
    print("IMAGE DEPENDENCE TEST")
    print("=" * 50)
    print(
        f"Normal output mean:   {normal_outputs.mean():.3f} ± {normal_outputs.std():.3f}"
    )
    print(
        f"Blackout output mean: {blackout_outputs.mean():.3f} ± {blackout_outputs.std():.3f}"
    )
    print(
        f"Shuffled output mean: {mask_only_outputs.mean():.3f} ± {mask_only_outputs.std():.3f}"
    )
    print(
        f"\nMean absolute difference (normal vs blackout): {blackout_diff:.4f}"
    )
    print(
        f"Mean absolute difference (normal vs shuffled): {shuffled_diff:.4f}"
    )
    print(f"\nCorrelation (normal vs blackout): {blackout_corr:.3f}")
    print(f"Correlation (normal vs shuffled): {shuffled_corr:.3f}")
    print("=" * 50)

    # Interpretation.
    if blackout_corr > 0.9:
        print("  WARNING: Model likely IGNORING image content!")
        print("  Predictions barely change when image is blacked out.")
        print("  Consider: increase cross_image_prob, check corruptions.")
    elif blackout_corr > 0.7:
        print("  CAUTION: Model may be under-utilizing image content.")
        print("  Predictions change significantly when image is blacked out.")
        print("  Consider: increase cross_image_prob, check corruptions.")
    else:
        print("  GOOD: Model appears to use image content.")
    print("=" * 50 + "\n")

    return {
        "blackout_diff": blackout_diff,
        "shuffled_diff": shuffled_diff,
        "blackout_corr": blackout_corr,
        "shuffled_corr": shuffled_corr,
    }


def main():
    args = parse_args()

    # Setup.
    set_seed(args.seed)
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    # Create output directory
    if args.experiment_name is None:
        args.experiment_name = f"mq_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Create corruption config.
    corruption_config = CorruptionConfig(
        identity_prob=args.identity_prob,
        cross_image_prob=args.cross_image_prob,
    )

    # Create dataloaders.
    print("\nCreating dataloaders...")
    if args.weak_checkpoint_paths:
        # Use extended dataloader that uses weak model predictions as corruption.
        from data.mq_dataset import get_mq_dataloaders_with_weak_models

        train_loader, val_loader, test_loader = (
            get_mq_dataloaders_with_weak_models(
                train_csv=args.train_csv,
                val_csv=args.val_csv,
                test_csv=args.test_csv,
                image_size=args.image_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                corruption_config=corruption_config,
                samples_per_image_train=args.samples_per_image,
                samples_per_image_val=args.samples_per_image_val,
                seed=args.seed,
                weak_checkpoint_paths=args.weak_checkpoint_paths,
                weak_model_prob=args.weak_model_prob,
                device="cuda",
            )
        )
    else:
        # Use standard dataloader.
        train_loader, val_loader, test_loader = get_mq_dataloaders(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            test_csv=args.test_csv,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            corruption_config=corruption_config,
            samples_per_image_train=args.samples_per_image,
            samples_per_image_val=1,  # Deterministic for validation
            seed=args.seed,
        )

    # Create model.
    print(f"\nCreating model with backbone: {args.backbone}")
    model = create_quality_predictor(
        backbone=args.backbone,
        pretrained=args.pretrained,
        dropout=args.dropout,
    )
    # model = model.to(device)

    # Count parameters.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # Loss and optimizer.
    if args.loss_func == "smoothl1":
        criterion = nn.SmoothL1Loss()
    elif args.loss_func == "mse":
        criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler.
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Restart every 10 epochs.
            T_mult=2,  # Double the period every restart.
            eta_min=1e-6,  # Minimum learning rate.
        )
    elif args.scheduler == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
        )

    # Early stopping.
    early_stopper = EarlyStopper(patience=args.patience)

    # Training loop.
    best_val_loss = float("inf")
    best_correlation = 0.0

    accelerator = Accelerator(
        mixed_precision="fp16",
        rng_types=["torch", "cuda", "generator"],
        log_with="comet_ml",
    )

    accelerator.init_trackers(
        project_name=args.comet_project,
        config=vars(args),
    )

    # Set up experiment tracking.
    experiment = None
    if args.use_comet and COMET_AVAILABLE:
        # experiment = Experiment(project_name=args.comet_project)
        experiment = accelerator.get_tracker("comet_ml").tracker
        experiment.set_name(args.experiment_name)
        experiment.log_parameters(vars(args))

    # Log arguments to Comet.
    if experiment:
        experiment.log_parameters(vars(args))

    model, optimizer, train_loader, val_loader, test_loader, scheduler = (
        accelerator.prepare(
            model, optimizer, train_loader, val_loader, test_loader, scheduler
        )
    )

    print("\nStarting training...")
    for epoch in range(args.epochs):
        # Train .
        train_loss, train_mae = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch, accelerator
        )

        # Validate.
        val_loss, val_mae, val_corr = validate(
            model, val_loader, criterion, epoch
        )

        # Update scheduler.
        if args.scheduler == "onecycle":
            # OneCycleLR steps per batch.
            scheduler.step()
        elif args.scheduler == "cosine":
            # CosineAnnealingLR steps per epoch.
            scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log metrics.
        print(f"\nEpoch {epoch + 1}/{args.epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}")
        print(
            f"  Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, Corr: {val_corr:.4f}"
        )
        print(f"  LR: {current_lr:.6f}")

        if experiment:
            experiment.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_mae": train_mae,
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "val_correlation": val_corr,
                    "lr": current_lr,
                },
                step=epoch,
            )

        # Save best model.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_correlation = val_corr

            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)

            checkpoint_path = output_dir / "best_model.pth"
            accelerator.save(unwrapped_model.state_dict(), checkpoint_path)
            # torch.save(
            #     {
            #         "epoch": epoch,
            #         "model_state_dict": unwrapped_model.state_dict(),
            #         "optimizer_state_dict": optimizer.state_dict(),
            #         "val_loss": val_loss,
            #         "val_correlation": val_corr,
            #         "args": vars(args),
            #     },
            #     checkpoint_path,
            # )
            print(f"  Saved best model to {checkpoint_path}")

        # Early stopping.
        if early_stopper.should_stop(val_loss):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Final evaluation on test set.
    if test_loader is not None:
        print("\nEvaluating on test set...")

        # Load best model
        # checkpoint = torch.load(
        #     output_dir / "best_model.pth", weights_only=False
        # )
        # model.load_state_dict(checkpoint["model_state_dict"])
        best_model_path = output_dir / "best_model.pth"
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(
            torch.load(best_model_path, map_location=accelerator.device)
        )

        test_loss, test_mae, test_corr = validate(
            model, test_loader, criterion, epoch=0
        )

        print(f"Test Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  MAE: {test_mae:.4f}")
        print(f"  Correlation: {test_corr:.4f}")

        if experiment:
            experiment.log_metrics(
                {
                    "test_loss": test_loss,
                    "test_mae": test_mae,
                    "test_correlation": test_corr,
                }
            )

    # Run image dependence test.
    test_image_dependence(model, test_loader, n_batches=10)

    # Save final summary.
    summary = {
        "best_val_loss": best_val_loss,
        "best_correlation": best_correlation,
        "total_epochs": epoch + 1,
        "args": vars(args),
    }

    import json

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best correlation: {best_correlation:.4f}")
    print(f"Checkpoints saved to: {output_dir}")

    if experiment:
        experiment.end()


if __name__ == "__main__":
    main()
