from typing import Dict

import torch

# =============================================================================
# UTILITIES: DICE COEFFICIENT, JACCARD INDEX (IoU), ACCURACY, F1 SCORE
# =============================================================================


def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
    compute_soft_dice: bool = False,
) -> torch.Tensor:
    """
    Compute Dice coefficient for evaluation.

    Args:
        pred: Predictions [B, 1, H, W] or [B, H, W]
        target: Ground truth [B, 1, H, W] or [B, H, W]
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor
        compute_soft_dice: If True, compute soft Dice coefficient

    Returns:
        Mean Dice coefficient
    """
    if not compute_soft_dice:
        pred = (pred > threshold).float()
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean()


def jaccard_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Jaccard Index (IoU) for binary segmentation.

    Args:
        pred: Predicted mask (after sigmoid), shape [B, 1, H, W]
        target: Ground truth mask, shape [B, 1, H, W]
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Mean Jaccard/IoU score across the batch
    """
    pred = (pred > threshold).float()

    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

    jaccard = (intersection + smooth) / (union + smooth)
    return jaccard.mean()


def pixel_accuracy(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """
    Pixel-wise accuracy for binary segmentation.

    Args:
        pred: Predicted mask (after sigmoid), shape [B, 1, H, W]
        target: Ground truth mask, shape [B, 1, H, W]
        threshold: Threshold for binarizing predictions

    Returns:
        Mean pixel accuracy across the batch
    """
    pred_binary = (pred > threshold).float()
    correct = (pred_binary == target).float()
    accuracy = correct.view(correct.size(0), -1).mean(dim=1)
    return accuracy.mean()


def f1_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    F1 Score (same as Dice for binary segmentation, but computed via
    precision/recall).

    Args:
        pred: Predicted mask (after sigmoid), shape [B, 1, H, W]
        target: Ground truth mask, shape [B, 1, H, W]
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Mean F1 score across the batch
    """
    pred_binary = (pred > threshold).float()
    pred_flat = pred_binary.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    tp = (pred_flat * target_flat).sum(dim=1)
    fp = (pred_flat * (1 - target_flat)).sum(dim=1)
    fn = ((1 - pred_flat) * target_flat).sum(dim=1)

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    f1 = 2 * (precision * recall) / (precision + recall + smooth)
    return f1.mean()


def f1_score_batch_level(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Batch-level (global) F1 score for binary segmentation. All pixels across
    the batch are pooled before computing precision/recall.


    Args:
        pred: Predicted mask (after sigmoid), shape [B, 1, H, W]
        target: Ground truth mask, shape [B, 1, H, W]
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor to avoid division by zero

    Returns:
        F1 score for each image in the batch
    """
    pred_binary = (pred > threshold).float()
    # Now, flatten everything. This means batch + spatial dimensions are
    # pooled.
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)

    # Now, we create GLOBAL confusion matrix.
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    f1 = 2 * (precision * recall) / (precision + recall + smooth)
    return f1


def compute_all_metrics(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute all metrics for binary segmentation at once.

    Args:
        pred: Predicted mask (after sigmoid), shape [B, 1, H, W]
        target: Ground truth mask, shape [B, 1, H, W]
        threshold: Threshold for binarizing predictions

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "dice": dice_coefficient(pred, target, threshold),
        "dice_soft": dice_coefficient(
            pred, target, threshold, compute_soft_dice=True
        ),
        "jaccard": jaccard_coefficient(pred, target, threshold),
        "pixel_accuracy": pixel_accuracy(pred, target, threshold),
        # "f1_score": f1_score(pred, target, threshold),
        "f1_score_batch_level": f1_score_batch_level(pred, target, threshold),
    }
    return metrics


# =============================================================================
# TESTING
# =============================================================================


def test_metrics():
    """
    Test metrics on a dummy dataset.
    """
    batch_size = 16
    h, w = 224, 224

    # Create dummy data.
    pred_logits = torch.randn(batch_size, 1, h, w)
    pred_probs = torch.sigmoid(pred_logits)
    target = (torch.rand(batch_size, 1, h, w) > 0.5).float()
    image = torch.randn(batch_size, 3, h, w)

    # Test Dice coefficient.
    dice = dice_coefficient(pred_probs, target)
    print(f"Dice coefficient: {dice.item():.4f}")

    # Test Dice coefficient (soft).
    dice_soft = dice_coefficient(pred_probs, target, compute_soft_dice=True)
    print(f"Dice coefficient (soft): {dice_soft.item():.4f}")

    # Test Jaccard coefficient.
    jaccard = jaccard_coefficient(pred_probs, target)
    print(f"Jaccard coefficient: {jaccard.item():.4f}")

    # Test pixel accuracy.
    accuracy = pixel_accuracy(pred_probs, target)
    print(f"Pixel accuracy: {accuracy.item():.4f}")

    # Test F1 score.
    f1 = f1_score(pred_probs, target)
    print(f"F1 score: {f1.item():.4f}")

    # Test F1 score (batch level).
    f1_batch = f1_score_batch_level(pred_probs, target)
    print(f"F1 score (batch level): {f1_batch.item():.4f}")
    pass


if __name__ == "__main__":
    test_metrics()
