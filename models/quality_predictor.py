"""
Segmentation Quality Predictor (g_φ) Model.

This model predicts the quality (Dice coefficient) of a segmentation mask
given the original image and the mask concatenated as a 4-channel input.
"""

from typing import Optional, Tuple

import timm
import torch
import torch.nn as nn


class QualityPredictor(nn.Module):
    """
    Segmentation Quality Predictor (g_φ).

    Takes a 4-channel input (RGB image + mask) and predicts a quality score ∈ [0, 1].

    Architecture:
    - Pretrained CNN backbone (e.g., EfficientNet) with modified first convolutional layer.
    - Global average pooling.
    - Regression head → single scalar output.
    """

    def __init__(
        self,
        backbone_name: str = "tf_efficientnetv2_s.in1k",
        pretrained: bool = True,
        dropout: float = 0.2,
        output_activation: str = "sigmoid",  # 'sigmoid', 'clamp', or 'none'
    ):
        """
        Initialize the Quality Predictor.

        Args:
            backbone_name: Name of the timm backbone model.
            pretrained: Whether to use pretrained weights.
            dropout: Dropout rate before final layer.
            output_activation: How to constrain output to [0, 1].
        """
        super().__init__()

        self.backbone_name = backbone_name
        self.output_activation = output_activation

        # Load pretrained backbone.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier, we add a regression head.
        )

        # Modify first conv layer to accept 4 channels.
        self._modify_first_conv()

        ## Get feature dimension.
        ## self.feature_dim = self.backbone.num_features
        ## Instead of using num_features, we dynamically get the feature
        ## dimension by running a dummy input through the backbone.
        ## Get feature dimension (query it dynamically to handle all backbones).
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 224, 224)
            dummy_features = self.backbone(dummy)
            self.feature_dim = dummy_features.shape[1]

        # Regression head.
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),
        )

        # Initialize head weights.
        self._init_head()

    def _modify_first_conv(self):
        """
        Modify the first conv layer to accept 4-channel input.
        """
        # Different backbones have different names for the first conv.
        first_conv_name = None
        first_conv = None

        # Try common names.
        for name in [
            "conv_stem",
            "conv1",
            "stem.conv",
            "features.0",
            "stem.0",
        ]:
            try:
                parts = name.split(".")
                module = self.backbone
                for part in parts:
                    if part.isdigit():
                        module = module[int(part)]
                    else:
                        module = getattr(module, part)

                if isinstance(module, nn.Conv2d):
                    first_conv_name = name
                    first_conv = module
                    break
            except (AttributeError, IndexError, TypeError):
                continue

        if first_conv is None:
            raise ValueError(
                f"Could not find first conv layer in {self.backbone_name}"
            )

        # Create new conv layer with 4 input channels.
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )

        # Initialize weights.
        with torch.no_grad():
            # Copy RGB weights.
            new_conv.weight[:, :3, :, :] = first_conv.weight
            # Initialize mask channel with average of RGB weights.
            new_conv.weight[:, 3:4, :, :] = first_conv.weight.mean(
                dim=1, keepdim=True
            )

            if first_conv.bias is not None:
                new_conv.bias = first_conv.bias

        # Replace the conv layer.
        parts = first_conv_name.split(".")
        module = self.backbone
        for part in parts[:-1]:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)

        if parts[-1].isdigit():
            module[int(parts[-1])] = new_conv
        else:
            setattr(module, parts[-1], new_conv)

    def _init_head(self):
        """
        Initialize regression head weights.
        """
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, 4, H, W] (RGB + mask).

        Returns:
            Quality scores of shape [B, 1].
        """
        # Extract features.
        features = self.backbone(x)

        # Regression head.
        output = self.head(features)

        # Apply output activation.
        if self.output_activation == "sigmoid":
            output = torch.sigmoid(output)
        elif self.output_activation == "clamp":
            output = torch.clamp(output, 0, 1)
        # else: no activation (for training with appropriate loss).

        return output

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without the regression head.

        Useful for transfer learning / encoder pretraining.
        """
        return self.backbone(x)


def create_quality_predictor(
    backbone: str = "tf_efficientnetv2_s.in1k",
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Factory function to create a Quality Predictor.

    Args:
        backbone: Name of the timm backbone
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate

    Returns:
        QualityPredictor
    """
    return QualityPredictor(
        backbone_name=backbone,
        pretrained=pretrained,
        dropout=dropout,
    )


# =============================================================================
# Available backbone options.
# =============================================================================

RECOMMENDED_BACKBONES = {
    # EfficientNet.
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    # ConvNeXt.
    "convnext_tiny": "convnext_tiny.fb_in1k",
    # ResNet.
    "resnet18": "resnet18.a1_in1k",
    # DenseNet.
    "densenet121": "densenet121.tv_in1k",
    # MobileNet.
    "mobilenet_v3": "mobilenetv3_large_100.ra_in1k",
}


def list_available_backbones():
    """
    Print available backbone options.
    """
    print("Recommended backbones for Quality Predictor:")
    print("-" * 50)
    for name, model_name in RECOMMENDED_BACKBONES.items():
        print(f"  {name}: {model_name}")
    print("-" * 50)
    print("You can also use any timm model name directly.")


# =============================================================================
# Testing.
# =============================================================================


def test_quality_predictor():
    """
    Test the quality predictor model.
    """
    print("Testing QualityPredictor...")

    # backbone = "tf_efficientnetv2_s.in1k"
    # backbone = "efficientnet_b0.ra_in1k"
    # backbone = "mobilenetv3_large_100.ra_in1k"
    # backbone = "densenet121.tv_in1k"
    # backbone = "convnext_tiny.in12k_ft_in1k"
    backbone = "resnet18.a1_in1k"

    # Create model.
    model = create_quality_predictor(
        backbone=backbone,
        pretrained=False,  # Skip download for testing.
    )

    # Test forward pass.
    batch_size = 4
    x = torch.randn(batch_size, 4, 224, 224)

    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(
        f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]"
    )

    # Test feature extraction.
    features = model.get_features(x)
    print(f"Feature shape: {features.shape}")

    # Count parameters.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_quality_predictor()
