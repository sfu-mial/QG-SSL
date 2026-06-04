"""
Models module.

Contains:
- quality_predictor: g_φ architecture for quality prediction.
- ema: Exponential moving average model wrapper.
- swin_unet: Swin-UNet (Swin Transformer-based U-Net) architecture
             for medical image segmentation.

Usage (Swin-UNet):
    from models import SwinUNet

    # Create model without pretrained weights.
    model = SwinUNet(img_size=224, num_classes=4, in_chans=3)

    # Create model with pretrained Swin Transformer weights.
    model = get_swin_unet_with_pretrained(num_classes=4, pretrained=True)

    # Or use the convenience function.
    model = swin_unet_tiny_patch4_window7_224(num_classes=4)
"""

from .ema import EMAModel, EMAModelWithRampup, create_ema_model
from .quality_predictor import (
    RECOMMENDED_BACKBONES,
    QualityPredictor,
    create_quality_predictor,
    list_available_backbones,
)
from .swin_unet import (
    SwinUNet,
    SwinUNet_Tiny,
    swin_unet_tiny_patch4_window7_224,
)
from .swin_unet_utils import (
    download_pretrained_weights,
    get_swin_unet_with_pretrained,
    load_pretrained_swin_unet,
)

__all__ = [
    # Quality predictor
    "QualityPredictor",
    "create_quality_predictor",
    "RECOMMENDED_BACKBONES",
    "list_available_backbones",
    # EMA
    "EMAModel",
    "EMAModelWithRampup",
    "create_ema_model",
    # Main model class for Swin-UNet
    "SwinUNet",
    # Factory functions for Swin-UNet
    "swin_unet_tiny_patch4_window7_224",
    "SwinUNet_Tiny",
    "get_swin_unet_with_pretrained",
    # Utilities for Swin-UNet
    "load_pretrained_swin_unet",
    "download_pretrained_weights",
]
