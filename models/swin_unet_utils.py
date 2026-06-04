"""
Utilities for loading pretrained weights for Swin-UNet.

The official Swin-UNet uses pretrained Swin Transformer weights from:
https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth

Usage:
    from models.swin_unet_utils import load_pretrained_swin_unet

    model = SwinUNet(img_size=224, num_classes=4, in_chans=3)
    load_pretrained_swin_unet(model, pretrained_path="path/to/swin_tiny_patch4_window7_224.pth")
"""

import os
import urllib.request
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# URLs for pretrained weights.
PRETRAINED_URLS = {
    "swin_tiny_patch4_window7_224": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
    "swin_tiny_patch4_window7_224_22k": "https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth",
}


def download_pretrained_weights(
    model_name: str = "swin_tiny_patch4_window7_224",
    save_dir: str = "./pretrained_ckpt",
) -> str:
    """
    Download pretrained Swin Transformer weights.

    Args:
        model_name: Name of the pretrained model.
        save_dir: Directory to save the weights.

    Returns:
        Path to the downloaded weights file.
    """
    if model_name not in PRETRAINED_URLS:
        raise ValueError(
            f"Unknown model name: {model_name}. Available: {list(PRETRAINED_URLS.keys())}"
        )

    url = PRETRAINED_URLS[model_name]
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = url.split("/")[-1]
    save_path = save_dir / filename

    if save_path.exists():
        print(f"Pretrained weights already exist at {save_path}")
        return str(save_path)

    print(f"Downloading pretrained weights from {url}...")
    urllib.request.urlretrieve(url, save_path)
    print(f"Saved to {save_path}")

    return str(save_path)


def load_pretrained_swin_unet(
    model: nn.Module,
    pretrained_path: Optional[str] = None,
    model_name: str = "swin_tiny_patch4_window7_224",
    download: bool = True,
    strict: bool = False,
) -> nn.Module:
    """
    Load pretrained Swin Transformer weights into Swin-UNet model.

    The pretrained weights are from the original Swin Transformer trained on ImageNet.
    They will be loaded into the encoder part of Swin-UNet. The decoder will use
    random initialization or you can provide full Swin-UNet weights.

    Args:
        model: SwinUNet model instance.
        pretrained_path: Path to pretrained weights. If None and download=True,
                        weights will be downloaded automatically.
        model_name: Name of pretrained model (used for automatic download).
        download: Whether to download weights if pretrained_path is None.
        strict: Whether to strictly enforce that the keys in state_dict match.

    Returns:
        Model with loaded weights.
    """
    if pretrained_path is None:
        if download:
            pretrained_path = download_pretrained_weights(model_name)
        else:
            raise ValueError("pretrained_path is None and download=False")

    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(
            f"Pretrained weights not found at {pretrained_path}"
        )

    print(f"Loading pretrained weights from {pretrained_path}")

    # Load checkpoint.
    checkpoint = torch.load(pretrained_path, map_location="cpu")

    # Handle different checkpoint formats.
    if "model" in checkpoint:
        pretrained_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        pretrained_dict = checkpoint["state_dict"]
    else:
        pretrained_dict = checkpoint

    # Get model state dict.
    model_dict = model.state_dict()

    # Filter and match weights.
    matched_dict = {}
    unmatched_keys = []

    for k, v in pretrained_dict.items():
        # Handle different naming conventions.
        # Swin Transformer uses 'layers.X' while Swin-UNet uses the same.
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                matched_dict[k] = v
            else:
                unmatched_keys.append(
                    (k, f"shape mismatch: {v.shape} vs {model_dict[k].shape}")
                )
        else:
            # Try to find a matching key with slightly different naming.
            matched = False
            for model_key in model_dict.keys():
                if k.replace(
                    "layers.", "layers."
                ) in model_key or model_key.endswith(k.split(".")[-1]):
                    if model_dict[model_key].shape == v.shape:
                        matched_dict[model_key] = v
                        matched = True
                        break
            if not matched:
                unmatched_keys.append((k, "key not found"))

    # Load matched weights.
    model_dict.update(matched_dict)

    # Report loading statistics.
    n_loaded = len(matched_dict)
    n_total = len(model_dict)
    n_pretrained = len(pretrained_dict)

    print(
        f"Loaded {n_loaded}/{n_total} layers from pretrained checkpoint ({n_pretrained} available)"
    )

    if unmatched_keys and len(unmatched_keys) <= 20:
        print(f"Unmatched keys ({len(unmatched_keys)}):")
        for k, reason in unmatched_keys[:10]:
            print(f"  - {k}: {reason}")
        if len(unmatched_keys) > 10:
            print(f"  ... and {len(unmatched_keys) - 10} more")

    # Load state dict.
    model.load_state_dict(model_dict, strict=strict)

    return model


def get_swin_unet_with_pretrained(
    num_classes: int = 1,
    in_chans: int = 3,
    img_size: int = 224,
    pretrained: bool = True,
    pretrained_path: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    """
    Create a Swin-UNet model with pretrained encoder weights.

    This is a convenience function that creates a Swin-UNet model and loads
    pretrained Swin Transformer weights.

    Args:
        num_classes: Number of output segmentation classes.
        in_chans: Number of input channels.
        img_size: Input image size.
        pretrained: Whether to load pretrained weights.
        pretrained_path: Path to pretrained weights (optional).
        **kwargs: Additional arguments passed to SwinUNet.

    Returns:
        SwinUNet model with pretrained weights (if requested).

    Example:
        >>> model = get_swin_unet_with_pretrained(num_classes=4, pretrained=True)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> y = model(x)  # (1, 4, 224, 224)
    """
    from .swin_unet import SwinUNet

    model = SwinUNet(
        img_size=img_size,
        num_classes=num_classes,
        in_chans=in_chans,
        **kwargs,
    )

    if pretrained:
        load_pretrained_swin_unet(model, pretrained_path=pretrained_path)

    return model


if __name__ == "__main__":
    # Test downloading and loading.
    from swin_unet import SwinUNet

    model = SwinUNet(img_size=224, num_classes=4, in_chans=3)
    print(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Test download (uncomment to actually download).
    # load_pretrained_swin_unet(model, download=True)
