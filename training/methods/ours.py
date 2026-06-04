import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from accelerate import Accelerator
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.ema import EMAModel
from training.losses import (
    BCEDiceLoss,
    PerSampleBCEDiceLoss,
    QualityGuidedLoss,
)
from training.metrics import dice_coefficient

# print("Testing imports.")
# exit()


def train_ours_qar(
    model: nn.Module,
    quality_predictor: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    optimizer: optim.Optimizer,
    lambda_unsup: float,
    accelerator: Accelerator,
    epoch: int,
    grad_clip: float = 1.0,
    ema_model: Optional[EMAModel] = None,
) -> Dict[str, float]:
    """
    QAR: Quality-aware regularization loss.
    Uses g_φ on unlabeled data to regularize the segmentation model's output.

    Args:
        model: Segmentation model f_θ to train.
        quality_predictor: Frozen g_φ that predicts segmentation quality.
        labeled_loader: DataLoader for labeled data.
        unlabeled_loader: DataLoader for unlabeled data.
        optimizer: Optimizer for model.
        lambda_unsup: Weight for unsupervised loss.
        accelerator: HuggingFace Accelerator.
        epoch: Current epoch number.
        grad_clip: Gradient clipping value (0 to disable).
        ema_model: Optional EMA model for teacher.

    Returns:
        Dictionary of averaged metrics for the epoch.
    """
    model.train()
    criterion = BCEDiceLoss()
    quality_loss = QualityGuidedLoss(quality_predictor)

    metrics = {
        "sup_loss": 0,
        "unsup_loss": 0,
        "total_loss": 0,
        "dice": 0,
        "pred_quality": 0,
    }
    n_batches = 0

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    n_iters = max(len(labeled_loader), len(unlabeled_loader))

    pbar = tqdm(
        range(n_iters),
        desc=f"Epoch {epoch + 1} [Train]",
        disable=not accelerator.is_main_process,
    )
    for _ in pbar:
        # Labeled data D_L.
        try:
            images_l, masks_l = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            images_l, masks_l = next(labeled_iter)

        # Unlabeled data D_U.
        try:
            images_u = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            images_u = next(unlabeled_iter)

        # Supervised loss (Eqn. 4 in the paper).
        pred_l = model(images_l)
        l_sup = criterion(pred_l, masks_l)

        # Quality-guided unsupervised loss (Eqn. 5 in the paper).
        if lambda_unsup > 0:
            pred_u = model(images_u)
            pred_u_prob = torch.sigmoid(pred_u)
            l_unsup, pred_quality = quality_loss(
                images_u, pred_u_prob, return_quality=True
            )
        else:
            l_unsup = torch.tensor(0.0, device=accelerator.device)
            pred_quality = None

        # Total loss (Eqn. 6 in the paper).
        total_loss = l_sup + lambda_unsup * l_unsup

        # Backward pass and optimization.
        optimizer.zero_grad()
        accelerator.backward(total_loss)
        if grad_clip > 0:
            accelerator.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if ema_model is not None:
            unwrapped_model = accelerator.unwrap_model(model)
            ema_model.update(unwrapped_model)

        with torch.no_grad():
            dice = dice_coefficient(torch.sigmoid(pred_l), masks_l)

        metrics["sup_loss"] += l_sup.item()
        metrics["unsup_loss"] += (
            l_unsup.item() if torch.is_tensor(l_unsup) else l_unsup
        )
        metrics["total_loss"] += total_loss.item()
        metrics["dice"] += dice.item()
        if pred_quality is not None:
            metrics["pred_quality"] += pred_quality.mean().item()
        n_batches += 1

        if accelerator.is_main_process:
            pbar.set_postfix(
                sup=f"{l_sup.item():.3f}",
                qual=f"{pred_quality.mean().item() if pred_quality is not None else 0:.3f}",
            )

    return {k: v / n_batches for k, v in metrics.items()}


def train_ours_pl_qw(
    model: nn.Module,
    quality_predictor: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    optimizer: optim.Optimizer,
    lambda_unsup: float,
    accelerator: Accelerator,
    epoch: int,
    grad_clip: float = 1.0,
    ema_model: Optional[EMAModel] = None,
    quality_threshold: float = 0.0,
    quality_temperature: float = 1.0,
) -> Dict[str, float]:
    """
    PL-QW: Quality-weighted pseudo-labeling.
    Uses g_φ to weight the pseudo-labels obtained for unlabeled data.

    Instead of using g_φ output as a loss signal, we use it as a weight
    for the standard pseudo-label loss. High-quality pseudo-labels
    contribute more to training, low-quality ones are down-weighted.

    Args:
        model: Segmentation model f_θ to train.
        quality_predictor: Frozen g_φ that predicts segmentation quality.
        labeled_loader: DataLoader for labeled data.
        unlabeled_loader: DataLoader for unlabeled data.
        optimizer: Optimizer for model.
        lambda_unsup: Weight for unsupervised loss.
        accelerator: HuggingFace Accelerator.
        epoch: Current epoch number.
        grad_clip: Gradient clipping value (0 to disable).
        ema_model: Optional EMA model for teacher.
        quality_threshold: Minimum quality score to include sample (0-1).
        quality_temperature: Sharpening temperature for weights (>1 sharpens).

    Returns:
        Dictionary of averaged metrics for the epoch
    """
    model.train()
    quality_predictor.eval()  # g_φ is always frozen.

    criterion = BCEDiceLoss()
    per_sample_criterion = PerSampleBCEDiceLoss()

    metrics = {
        "sup_loss": 0,
        "unsup_loss": 0,
        "total_loss": 0,
        "dice": 0,
        "pred_quality": 0,
        "effective_samples": 0,  # Track how many samples pass threshold.
    }
    n_batches = 0

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    n_iters = max(len(labeled_loader), len(unlabeled_loader))

    pbar = tqdm(
        range(n_iters),
        desc=f"Epoch {epoch + 1} [Train]",
        disable=not accelerator.is_main_process,
    )

    for _ in pbar:
        # Labeled data D_L.
        try:
            images_l, masks_l = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            images_l, masks_l = next(labeled_iter)

        # Unlabeled data D_U.
        try:
            images_u = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            images_u = next(unlabeled_iter)

        # Supervised loss (Eqn. 4 in the paper).
        pred_l = model(images_l)
        l_sup = criterion(pred_l, masks_l)

        # Quality-weighted pseudo-label loss (Eqn. 7 in the paper).
        if lambda_unsup > 0:
            # Generate pseudo-labels (detached - no gradient through this).
            with torch.no_grad():
                pseudo_logits = ema_model(images_u)
                pseudo_probs = torch.sigmoid(pseudo_logits)
                pseudo_labels = (pseudo_probs > 0.5).float()

            # Get quality scores from g_φ (frozen, no gradient).
            with torch.no_grad():
                # g_φ expects (image, mask) and outputs quality score in [0, 1].
                quality_input = torch.cat(
                    [images_u, pseudo_labels], dim=1
                ).float()
                quality_scores = quality_predictor(quality_input)
                quality_scores = quality_scores.squeeze()  # (B,)

                # Apply threshold - create mask for samples above threshold.
                valid_mask = quality_scores >= quality_threshold

                # Apply temperature sharpening: weight = quality^temperature.
                # temperature > 1 sharpens (high quality weighted more).
                # temperature < 1 softens (more uniform weights).
                weights = quality_scores**quality_temperature

                # Zero out weights for samples below threshold.
                weights = weights * valid_mask.float()

            # Forward pass for unlabeled (this one needs gradients).
            pred_u = model(images_u)

            # Compute per-sample loss.
            # BCEDiceLoss typically reduces over batch, we need per-sample.
            pred_u_prob = torch.sigmoid(pred_u)
            per_sample_loss = per_sample_criterion(pred_u_prob, pseudo_labels)

            # Weighted average of losses.
            # Eqn. 7 in the paper.
            if weights.sum() > 0:
                l_unsup = (weights * per_sample_loss).sum() / (
                    weights.sum() + 1e-8
                )
            else:
                l_unsup = torch.tensor(0.0, device=accelerator.device)

            pred_quality = quality_scores
            effective = valid_mask.float().mean()
        else:
            l_unsup = torch.tensor(0.0, device=accelerator.device)
            pred_quality = None
            effective = 0.0

        # Optimization step.
        # Total loss (Eqn. 8 in the paper).
        total_loss = l_sup + lambda_unsup * l_unsup

        optimizer.zero_grad()
        accelerator.backward(total_loss)
        if grad_clip > 0:
            accelerator.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # EMA update.
        if ema_model is not None:
            unwrapped_model = accelerator.unwrap_model(model)
            ema_model.update(unwrapped_model)

        # Metrics.
        with torch.no_grad():
            dice = dice_coefficient(torch.sigmoid(pred_l), masks_l)

        metrics["sup_loss"] += l_sup.item()
        metrics["unsup_loss"] += (
            l_unsup.item() if torch.is_tensor(l_unsup) else l_unsup
        )
        metrics["total_loss"] += total_loss.item()
        metrics["dice"] += dice.item()
        if pred_quality is not None:
            metrics["pred_quality"] += pred_quality.mean().item()
        metrics["effective_samples"] += (
            effective.item() if torch.is_tensor(effective) else effective
        )
        n_batches += 1

        if accelerator.is_main_process:
            pbar.set_postfix(
                sup=f"{l_sup.item():.3f}",
                unsup=f"{l_unsup.item() if torch.is_tensor(l_unsup) else 0:.3f}",
                qual=f"{pred_quality.mean().item() if pred_quality is not None else 0:.3f}",
                eff=f"{effective.item() if torch.is_tensor(effective) else 0:.1%}",
            )

    return {k: v / n_batches for k, v in metrics.items()}
