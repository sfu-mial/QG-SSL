import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from .losses import BCEDiceLoss
from .metrics import compute_all_metrics, dice_coefficient


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    accelerator: Accelerator,
    epoch: int,
    quality_predictor: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """
    Validation loop for semi-supervised segmentation.

    Args:
        model: Segmentation model f_θ to validate.
        dataloader: Validation dataloader.
        accelerator: HuggingFace Accelerator.
        epoch: Current epoch number.
        quality_predictor: Optional g_φ for quality prediction metrics.

    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    criterion = BCEDiceLoss()

    metrics = {"val_loss": 0, "val_dice": 0}
    if quality_predictor is not None:
        metrics["val_pred_quality"] = 0
        metrics["val_alignment"] = 0

    all_pred_q, all_actual_dice = [], []
    n_batches = 0

    for images, masks in tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1} [Val]",
        disable=not accelerator.is_main_process,
    ):
        pred = model(images)
        pred_prob = torch.sigmoid(pred)

        loss = criterion(pred, masks)
        dice = dice_coefficient(pred_prob, masks)

        metrics["val_loss"] += loss.item()
        metrics["val_dice"] += dice.item()

        if quality_predictor is not None:
            quality_input = torch.cat([images, pred_prob], dim=1)
            pred_q = quality_predictor(quality_input)
            metrics["val_pred_quality"] += pred_q.mean().item()

            for i in range(pred_prob.size(0)):
                d = dice_coefficient(
                    pred_prob[i : i + 1], masks[i : i + 1]
                ).item()
                all_pred_q.append(pred_q[i].item())
                all_actual_dice.append(d)

        n_batches += 1

    # Create a dict to store the final metrics as key-value pairs.
    metrics = {k: v / n_batches for k, v in metrics.items()}

    # Compute the correlation between predicted quality and actual dice.
    if quality_predictor is not None and len(all_pred_q) > 10:
        pred_q_arr = np.array(all_pred_q)
        actual_arr = np.array(all_actual_dice)
        if pred_q_arr.std() > 1e-6 and actual_arr.std() > 1e-6:
            metrics["val_alignment"] = np.corrcoef(pred_q_arr, actual_arr)[
                0, 1
            ]

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    accelerator: Accelerator,
    quality_predictor: Optional[nn.Module] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Full evaluation with per-sample metrics.

    Computes all metrics (dice, jaccard, pixel_accuracy, f1_score, etc.)
    for each sample individually, then returns mean and std for each metric.

    Args:
        model: Segmentation model to evaluate
        dataloader: Test dataloader
        accelerator: HuggingFace Accelerator
        quality_predictor: Optional g_φ for quality prediction metrics

    Returns:
        Dict with structure: {metric_name: {"mean": x, "std": y}}
    """
    model.eval()

    # Initialize collectors for all metrics.
    metric_names = [
        "dice",
        "dice_soft",
        "jaccard",
        "pixel_accuracy",
        # "f1_score",
        "f1_score_batch_level",
    ]
    all_metrics = {name: [] for name in metric_names}

    # Additional metrics if quality predictor is available.
    if quality_predictor is not None:
        all_metrics["pred_quality"] = []
        all_metrics["quality_error"] = []  # |pred_quality - actual_dice|

    for images, masks in tqdm(
        dataloader,
        desc="Evaluating",
        disable=not accelerator.is_main_process,
    ):
        pred = model(images)
        pred_prob = torch.sigmoid(pred)

        # Compute per-sample metrics.
        batch_size = pred_prob.size(0)
        for i in range(batch_size):
            sample_pred = pred_prob[i : i + 1]
            sample_mask = masks[i : i + 1]

            # Get all metrics for this sample.
            sample_metrics = compute_all_metrics(sample_pred, sample_mask)

            for name in metric_names:
                value = sample_metrics[name]
                # Handle both tensor and float.
                if torch.is_tensor(value):
                    value = value.item()
                all_metrics[name].append(value)

            # Quality predictor metrics.
            if quality_predictor is not None:
                quality_input = torch.cat(
                    [images[i : i + 1], sample_pred], dim=1
                )
                pred_q = quality_predictor(quality_input).item()
                actual_dice = sample_metrics["dice"]
                if torch.is_tensor(actual_dice):
                    actual_dice = actual_dice.item()

                all_metrics["pred_quality"].append(pred_q)
                all_metrics["quality_error"].append(abs(pred_q - actual_dice))

    # Compute mean and std for each metric.
    results = {}
    for name, values in all_metrics.items():
        values_arr = np.array(values)
        results[name] = {
            "mean": float(np.mean(values_arr)),
            "std": float(np.std(values_arr)),
            "n_samples": len(values),
        }

    # Add correlation between predicted quality and actual Dice.
    if quality_predictor is not None and len(all_metrics["pred_quality"]) > 10:
        pred_q_arr = np.array(all_metrics["pred_quality"])
        dice_arr = np.array(all_metrics["dice"])
        if pred_q_arr.std() > 1e-6 and dice_arr.std() > 1e-6:
            correlation = np.corrcoef(pred_q_arr, dice_arr)[0, 1]
            results["quality_dice_correlation"] = {
                "mean": float(correlation),
                "std": 0.0,  # Single value, no std.
                "n_samples": len(pred_q_arr),
            }

    return results
