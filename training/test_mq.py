"""
Testing script for the Segmentation Quality Predictor (g_φ).

Loads a trained checkpoint and evaluates on a test set.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from data.mq_dataset import get_mq_dataloaders
from data.vqm_generator import CorruptionConfig
from models.quality_predictor import create_quality_predictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Segmentation Quality Predictor (g_φ)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--test_csv", type=str, required=True, help="Path to test CSV"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--samples_per_image", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


@torch.no_grad()
def evaluate(model, dataloader, criterion, accelerator=None):
    """
    Evaluate the model on a dataloader.

    Args:
        model: The g_φ model.
        dataloader: Test dataloader.
        criterion: Loss function.
        accelerator: Optional HuggingFace Accelerator.

    Returns:
        Dictionary with loss, mae, correlation, preds, targets.
    """
    model.eval()
    total_loss = 0.0
    # total_mae = 0.0
    n_batches = 0

    all_preds = []
    all_targets = []

    for inputs, targets in tqdm(dataloader, desc="Evaluating"):
        # inputs = inputs.to(device)
        # targets = targets.to(device).unsqueeze(1)
        targets = targets.unsqueeze(1)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # mae = torch.abs(outputs - targets).mean()

        total_loss += loss.item()
        # total_mae += mae.item()
        n_batches += 1

        all_preds.extend(outputs.cpu().numpy().flatten())
        all_targets.extend(targets.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate per-sample errors.
    abs_errors = np.abs(all_preds - all_targets)
    sq_errors = (all_preds - all_targets) ** 2

    correlation = 0.0
    if all_preds.std() > 1e-6 and all_targets.std() > 1e-6:
        correlation = np.corrcoef(all_preds, all_targets)[0, 1]

    return {
        "loss": total_loss / n_batches,
        # "mae": total_mae / n_batches,
        "mae_mean": abs_errors.mean(),
        "mae_std": abs_errors.std(),
        "mse_mean": sq_errors.mean(),
        "mse_std": sq_errors.std(),
        "correlation": correlation,
        "preds": all_preds,
        "targets": all_targets,
    }


@torch.no_grad()
def test_image_dependence(model, dataloader, n_batches=10, accelerator=None):
    """
    Test whether g_φ actually uses image content.

    This test checks if predictions change when:
    1. Image is blacked out (set to zero)
    2. Image is shuffled (wrong image paired with mask)

    A well-trained g_φ should produce different predictions when the image
    is modified, since it needs to compare image content against mask structure.

    Args:
        model: The g_φ model.
        dataloader: Test dataloader.
        n_batches: Number of batches to test.
        accelerator: Optional HuggingFace Accelerator.

    Returns:
        Dictionary with dependence metrics.
    """
    model.eval()

    normal_outputs = []
    blackout_outputs = []
    shuffled_outputs = []

    for i, (inputs, _) in enumerate(dataloader):
        if i >= n_batches:
            break

        # inputs = inputs.to(device)

        # Normal prediction (forward pass)
        out_normal = model(inputs)
        normal_outputs.extend(out_normal.cpu().numpy().flatten())

        # Blackout: set image's RGB channels to zero, keep mask unchanged.
        inputs_blackout = inputs.clone()
        inputs_blackout[:, :3, :, :] = 0
        out_blackout = model(inputs_blackout)
        blackout_outputs.extend(out_blackout.cpu().numpy().flatten())

        # Shuffled: permute images within batch, keep masks unchanged.
        inputs_shuffled = inputs.clone()
        perm = torch.randperm(inputs.size(0))
        inputs_shuffled[:, :3, :, :] = inputs[perm, :3, :, :]
        out_shuffled = model(inputs_shuffled)
        shuffled_outputs.extend(out_shuffled.cpu().numpy().flatten())

    normal_outputs = np.array(normal_outputs)
    blackout_outputs = np.array(blackout_outputs)
    shuffled_outputs = np.array(shuffled_outputs)

    # Compute differences and correlations
    blackout_diff = np.abs(normal_outputs - blackout_outputs).mean()
    shuffled_diff = np.abs(normal_outputs - shuffled_outputs).mean()
    blackout_corr = np.corrcoef(normal_outputs, blackout_outputs)[0, 1]
    shuffled_corr = np.corrcoef(normal_outputs, shuffled_outputs)[0, 1]

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
        f"Shuffled output mean: {shuffled_outputs.mean():.3f} ± {shuffled_outputs.std():.3f}"
    )
    print(f"\nMean abs diff (normal vs blackout): {blackout_diff:.4f}")
    print(f"Mean abs diff (normal vs shuffled): {shuffled_diff:.4f}")
    print(f"\nCorrelation (normal vs blackout): {blackout_corr:.3f}")
    print(f"Correlation (normal vs shuffled): {shuffled_corr:.3f}")
    print("=" * 50)

    if blackout_corr > 0.9:
        print("  WARNING: Model likely IGNORING image content!")
    elif blackout_corr > 0.7:
        print("  CAUTION: Model may be under-utilizing image content.")
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Use accelerate for device management
    from accelerate import Accelerator

    accelerator = Accelerator()
    print(f"Using device: {accelerator.device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(
        args.checkpoint, map_location=accelerator.device, weights_only=False
    )
    # saved_args = checkpoint.get("args", {})
    # Handle different checkpoint formats
    if "args" in checkpoint:
        saved_args = checkpoint["args"]
    else:
        saved_args = {}

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"  Trained for {checkpoint.get('epoch', '?') + 1} epochs")
        print(f"  Val loss: {checkpoint.get('val_loss', '?'):.4f}")
        print(
            f"  Val correlation: {checkpoint.get('val_correlation', '?'):.4f}"
        )
    else:
        # Assume it's just the state dict (accelerate saved format)
        state_dict = checkpoint
        print("  (Checkpoint saved in accelerate format)")

    # Create model
    model = create_quality_predictor(
        # backbone=saved_args.get("backbone", "tf_efficientnetv2_s.in1k"),
        backbone=saved_args.get("backbone", "resnet18.a1_in1k"),
        pretrained=False,
        dropout=saved_args.get("dropout", 0.2),
    )
    model.load_state_dict(state_dict)
    # model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Create dataloader (standard VQM generation, no weak models needed for testing).
    corruption_config = CorruptionConfig(
        identity_prob=saved_args.get("identity_prob", 0.12),
        cross_image_prob=saved_args.get("cross_image_prob", 0.18),
    )

    _, _, test_loader = get_mq_dataloaders(
        train_csv=args.test_csv,  # dummy, won't be used
        val_csv=args.test_csv,  # dummy, won't be used
        test_csv=args.test_csv,
        image_size=saved_args.get("image_size", 224),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        corruption_config=corruption_config,
        samples_per_image_train=1,
        samples_per_image_val=args.samples_per_image,
        seed=args.seed,
    )

    # Prepare with accelerator.
    model, test_loader = accelerator.prepare(model, test_loader)

    print(f"\nTest set: {len(test_loader.dataset)} samples")

    # Evaluate.
    criterion = nn.SmoothL1Loss()
    results = evaluate(model, test_loader, criterion, accelerator)

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"Loss:        {results['loss']:.4f}")
    print(f"MAE:         {results['mae_mean']:.4f} ± {results['mae_std']:.4f}")
    print(f"MSE:         {results['mse_mean']:.4f} ± {results['mse_std']:.4f}")
    print(f"Correlation: {results['correlation']:.4f}")
    print("=" * 50)

    # Image dependence test.
    dependence = test_image_dependence(
        model, test_loader, n_batches=10, accelerator=accelerator
    )

    # Summary.
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    quality_ok = results["correlation"] > 0.7
    image_ok = dependence["blackout_corr"] < 0.7

    print(
        f"Quality prediction:  {' GOOD' if quality_ok else ' POOR'} (corr={results['correlation']:.3f}, need >0.7)"
    )
    print(
        f"Image dependence:    {' GOOD' if image_ok else ' POOR'} (blackout_corr={dependence['blackout_corr']:.3f}, need <0.7)"
    )

    if quality_ok and image_ok:
        print("\n Model is ready for semi-supervised training!")
    else:
        print("\n Model needs improvement before semi-supervised training.")
        if not quality_ok:
            print(
                "  - Try: smaller backbone, lower dropout, more epochs, adjust LR"
            )
        if not image_ok:
            print(
                "  - Try: increase cross_image_prob, verify corruptions are diverse"
            )
    print("=" * 50)


if __name__ == "__main__":
    main()
