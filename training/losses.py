"""
Loss functions for semi-supervised segmentation training.

Includes:
- Standard segmentation losses (Dice, BCE)
- Quality-guided unsupervised loss
- Combined losses with lambda scheduling
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# SEGMENTATION LOSSES
# =============================================================================


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.

    Loss = 1 - Dice Coefficient
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            pred: Predictions [B, 1, H, W] or [B, H, W], values in [0, 1]
            target: Ground truth [B, 1, H, W] or [B, H, W], binary

        Returns:
            Scalar loss value
        """
        # Flatten spatial dimensions
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice loss.

    Often provides better gradients than Dice alone.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(
        self, pred_logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_logits: Raw logits [B, 1, H, W]
            target: Ground truth [B, 1, H, W], binary
        """
        bce_loss = self.bce(pred_logits, target)
        dice_loss = self.dice(torch.sigmoid(pred_logits), target)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p) = -α(1-p)^γ log(p)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self, pred_logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            pred_logits, target, reduction="none"
        )
        pred_prob = torch.sigmoid(pred_logits)

        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        return (focal_weight * bce).mean()


class PerSampleBCEDiceLoss(nn.Module):
    """
    Per-sample BCE + Dice loss.

    Args:
        bce_weight: Weight for BCE component.
        dice_weight: Weight for Dice component.
        smooth: Smoothing parameter for Dice loss.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-sample BCE + Dice loss.

        Args:
            pred: Predictions after sigmoid, shape (B, 1, H, W) or (B, H, W).
            target: Target masks, shape (B, 1, H, W) or (B, H, W).
            bce_weight: Weight for BCE component (1 - bce_weight for Dice).
            smooth: Smoothing factor for Dice.

        Returns:
            Per-sample loss tensor of shape (B,).
        """
        # Ensure 4D input.

        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        batch_size = pred.size(0)

        # Flatten spatial dimensions for each sample.
        pred_flat = pred.view(batch_size, -1)
        target_flat = target.view(batch_size, -1)

        # Per-sample BCE.
        bce = F.binary_cross_entropy(pred_flat, target_flat, reduction="none")
        bce_per_sample = bce.mean(dim=1)  # (B,)

        # Per-sample Dice loss.
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss_per_sample = 1.0 - dice_score  # (B,)

        # Combined loss.
        combined = (
            self.bce_weight * bce_per_sample
            + self.dice_weight * dice_loss_per_sample
        )

        return combined


# =============================================================================
# QUALITY-GUIDED LOSS
# =============================================================================


class QualityGuidedLoss(nn.Module):
    """
    Quality-guided unsupervised loss for semi-supervised learning.

    Uses the frozen Quality Predictor (g_φ) to evaluate segmentation
    predictions on unlabeled data and provide gradient signal.

    L_unsup = 1 - g_φ(x ⊕ f_θ(x))

    Maximizing predicted quality should improve segmentation.
    """

    def __init__(
        self,
        quality_predictor: nn.Module,
        gradient_clip: Optional[float] = 1.0,
        confidence_threshold: Optional[float] = None,
        use_soft_mask: bool = True,
    ):
        """
        Args:
            quality_predictor: Frozen g_φ model
            gradient_clip: Max gradient norm for stability (None to disable)
            confidence_threshold: Only apply loss if predicted quality > threshold
            use_soft_mask: If True, use soft predictions; if False, use hard (thresholded)
        """
        super().__init__()
        self.quality_predictor = quality_predictor
        self.gradient_clip = gradient_clip
        self.confidence_threshold = confidence_threshold
        self.use_soft_mask = use_soft_mask

        # Ensure quality predictor (g_φ) is frozen.
        for param in self.quality_predictor.parameters():
            param.requires_grad = False
        self.quality_predictor.eval()  # g_φ is always frozen.

    def forward(
        self,
        image: torch.Tensor,
        pred_mask: torch.Tensor,
        return_quality: bool = False,
    ) -> torch.Tensor:
        """
        Compute quality-guided loss.

        Args:
            image: Input images [B, 3, H, W]
            pred_mask: Predicted masks [B, 1, H, W], values in [0, 1]
            return_quality: If True, also return predicted quality scores

        Returns:
            Loss value (and optionally quality scores)
        """
        # Prepare mask.
        if self.use_soft_mask:
            mask_input = pred_mask
        else:
            mask_input = (pred_mask > 0.5).float()

        # Concatenate image and mask for quality predictor.
        # Quality predictor expects [B, 4, H, W].
        quality_input = torch.cat([image, mask_input], dim=1)

        # Get predicted quality.
        with torch.enable_grad():  # Ensure gradients flow through pred_mask.
            predicted_quality = self.quality_predictor(quality_input)

        # Loss = 1 - quality (we want to maximize quality).
        loss = 1 - predicted_quality.mean()

        # Optional: Apply confidence threshold.
        if self.confidence_threshold is not None:
            # Only count samples where quality predictor is confident.
            confident_mask = predicted_quality > self.confidence_threshold
            if confident_mask.sum() > 0:
                loss = (1 - predicted_quality[confident_mask]).mean()
            else:
                loss = torch.tensor(0.0, device=image.device)

        if return_quality:
            return loss, predicted_quality.detach()
        return loss

    def get_quality_scores(
        self,
        image: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get quality scores without computing loss.

        Args:
            image: Input images [B, 3, H, W]
            pred_mask: Predicted masks [B, 1, H, W], values in [0, 1]

        Returns:
            Predicted quality scores [B,]
        """
        with torch.no_grad():
            mask_input = (
                pred_mask if self.use_soft_mask else (pred_mask > 0.5).float()
            )
            quality_input = torch.cat([image, mask_input], dim=1)
            return self.quality_predictor(quality_input)


# =============================================================================
# TESTING
# =============================================================================


def test_losses():
    """Test loss functions."""
    batch_size = 4
    h, w = 224, 224

    # Create dummy data
    pred_logits = torch.randn(batch_size, 1, h, w)
    pred_probs = torch.sigmoid(pred_logits)
    target = (torch.rand(batch_size, 1, h, w) > 0.5).float()
    image = torch.randn(batch_size, 3, h, w)

    # Test Dice loss
    dice_loss = DiceLoss()
    loss = dice_loss(pred_probs, target)
    print(f"Dice loss: {loss.item():.4f}")

    # Test BCE + Dice loss
    bce_dice_loss = BCEDiceLoss()
    loss = bce_dice_loss(pred_logits, target)
    print(f"BCE + Dice loss: {loss.item():.4f}")

    # Test Focal loss
    focal_loss = FocalLoss()
    loss = focal_loss(pred_logits, target)
    print(f"Focal loss: {loss.item():.4f}")

    print("\nAll loss tests passed!")


if __name__ == "__main__":
    test_losses()
