"""
Swin-UNet: Unet-like Pure Transformer for Medical Image Segmentation
Based on: https://arxiv.org/abs/2105.05537 (ECCVW 2022)

This is a clean, self-contained implementation designed to be compatible with
segmentation_models_pytorch (SMP)-style usage:

    from models.swin_unet import SwinUNet
    model = SwinUNet(img_size=224, num_classes=4, in_chans=3)

Pretrained weights can be loaded from the official Swin Transformer checkpoints.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition into non-overlapping windows with padding if needed.

    Args:
        x: (B, H, W, C).
        window_size: window size.

    Returns:
        windows: (num_windows*B, window_size, window_size, C).
    """
    B, H, W, C = x.shape
    x = x.view(
        B, H // window_size, window_size, W // window_size, window_size, C
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(
    windows: torch.Tensor, window_size: int, H: int, W: int
) -> torch.Tensor:
    """
    Reverse window partition.

    Args:
        windows: (num_windows*B, window_size, window_size, C).
        window_size: Window size.
        H: Height of image.
        W: Width of image.

    Returns:
        x: (B, H, W, C).
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module with relative
    position bias.
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # Define a parameter table of relative position bias.
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads
            )
        )  # 2*Wh-1 * 2*Ww-1, nH

        # Get pair-wise relative position index for each token inside the window.
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += (
            self.window_size[0] - 1
        )  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer(
            "relative_position_index", relative_position_index
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: input features with shape of (num_windows*B, N, C).
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww)
                  or None.

        Returns:
            Output features with shape of (num_windows*B, N, C).
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(
                B_ // nW, nW, self.num_heads, N, N
            ) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # If window size is larger than input resolution, we don't partition windows.
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, (
            "shift_size must be in [0, window_size)"
        )

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            # Calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size
            )
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, (
            f"input feature has wrong size, expected {H * W}, got {L}"
        )

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # nW*B, window_size*window_size, C

        # Merge windows
        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, C
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, H, W
        )  # B H' W' C

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """
    Patch Merging Layer (Downsampling).
    """

    def __init__(
        self,
        input_resolution: Tuple[int, int],
        dim: int,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, (
            f"input feature has wrong size, expected {H * W}, got {L}"
        )
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand(nn.Module):
    """
    Patch Expanding Layer (Upsampling).
    """

    def __init__(
        self,
        input_resolution: Tuple[int, int],
        dim: int,
        dim_scale: int = 2,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = (
            nn.Linear(dim, 2 * dim, bias=False)
            if dim_scale == 2
            else nn.Identity()
        )
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, (
            f"input feature has wrong size, expected {H * W}, got {L}"
        )

        x = x.view(B, H, W, C)
        x = x.view(B, H, W, 2, 2, C // 4)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    """
    Final Patch Expanding Layer (4x Upsampling).
    """

    def __init__(
        self,
        input_resolution: Tuple[int, int],
        dim: int,
        dim_scale: int = 4,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, (
            f"input feature has wrong size, expected {H * W}, got {L}"
        )

        x = x.view(B, H, W, C)
        x = x.view(
            B, H, W, self.dim_scale, self.dim_scale, C // (self.dim_scale**2)
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(
            B, H * self.dim_scale, W * self.dim_scale, C // (self.dim_scale**2)
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage (encoder).
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks.
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # Patch merging layer.
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class BasicLayer_up(nn.Module):
    """
    A basic Swin Transformer layer for one stage (decoder).
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        upsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks.
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # Patch expanding layer.
        if upsample is not None:
            self.upsample = PatchExpand(
                input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer
            )
        else:
            self.upsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinUNet(nn.Module):
    """
    Swin-UNet: Unet-like Pure Transformer for Medical Image Segmentation.

    This implementation provides a clean interface compatible with segmentation_models_pytorch.

    Args:
        img_size: Input image size (default: 224).
        patch_size: Patch size (default: 4).
        in_chans: Number of input channels (default: 3).
        num_classes: Number of segmentation classes (default: 1).
        embed_dim: Embedding dimension (default: 96).
        depths: Depth of each Swin Transformer stage (default: [2, 2, 6, 2]).
        depths_decoder: Depth of each decoder stage (default: [2, 2, 2, 2]).
        num_heads: Number of attention heads in each stage (default: [3, 6, 12, 24]).
        window_size: Window size (default: 7).
        mlp_ratio: Ratio of mlp hidden dim to embedding dim (default: 4.0).
        qkv_bias: If True, add a learnable bias to query, key, value (default: True).
        qk_scale: Override default qk scale of head_dim ** -0.5 if set (default: None).
        drop_rate: Dropout rate (default: 0.0).
        attn_drop_rate: Attention dropout rate (default: 0.0).
        drop_path_rate: Stochastic depth rate (default: 0.1).
        norm_layer: Normalization layer (default: nn.LayerNorm).
        ape: If True, add absolute position embedding (default: False).
        patch_norm: If True, add normalization after patch embedding (default: True).
        use_checkpoint: Whether to use checkpointing to save memory (default: False).
        final_upsample: Type of final upsampling (default: "expand_first").
        pretrained: Path to pretrained weights or None (default: None).
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        depths_decoder: List[int] = [2, 2, 2, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        final_upsample: str = "expand_first",
        pretrained: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # Split image into non-overlapping patches.
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Absolute position embedding.
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth.
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]

        # Build encoder layers.
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                downsample=PatchMerging
                if (i_layer < self.num_layers - 1)
                else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # Build decoder layers.
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = (
                nn.Linear(
                    2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                )
                if i_layer > 0
                else nn.Identity()
            )

            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    dim_scale=2,
                    norm_layer=norm_layer,
                )
            else:
                layer_up = BasicLayer_up(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths_decoder[i_layer],
                    num_heads=num_heads[self.num_layers - 1 - i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[: self.num_layers - 1 - i_layer]) : sum(
                            depths[: self.num_layers - i_layer]
                        )
                    ],
                    norm_layer=norm_layer,
                    upsample=PatchExpand
                    if (i_layer < self.num_layers - 1)
                    else None,
                    use_checkpoint=use_checkpoint,
                )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            self.up = FinalPatchExpand_X4(
                input_resolution=(
                    img_size // patch_size,
                    img_size // patch_size,
                ),
                dim_scale=4,
                dim=embed_dim,
            )
            self.output = nn.Conv2d(
                in_channels=embed_dim,
                out_channels=self.num_classes,
                kernel_size=1,
                bias=False,
            )

        self.apply(self._init_weights)

        # Load pretrained weights if provided.
        if pretrained is not None:
            self.load_pretrained(pretrained)

    def _init_weights(self, m):
        """
        Initialize weights.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_pretrained(self, pretrained_path: str):
        """
        Load pretrained weights from official Swin Transformer checkpoint.

        Args:
            pretrained_path: Path to pretrained weights.
        """
        if pretrained_path.endswith(".npz"):
            # Load from numpy format (official Google weights)
            self._load_from_npz(pretrained_path)
        else:
            # Load from PyTorch format
            self._load_from_pth(pretrained_path)

    def _load_from_pth(self, pretrained_path: str):
        """
        Load weights from PyTorch checkpoint.

        Args:
            pretrained_path: Path to PyTorch checkpoint.
        """
        pretrained_dict = torch.load(pretrained_path, map_location="cpu")
        if "model" in pretrained_dict:
            pretrained_dict = pretrained_dict["model"]
        elif "state_dict" in pretrained_dict:
            pretrained_dict = pretrained_dict["state_dict"]

        model_dict = self.state_dict()

        # Filter out unnecessary keys and handle size mismatches.
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }

        # Log loading info
        print(
            f"Loading pretrained weights: {len(pretrained_dict)}/{len(model_dict)} layers matched"
        )

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)

    def _load_from_npz(self, pretrained_path: str):
        """
        Load weights from numpy checkpoint (for Google's pretrained weights).

        Args:
            pretrained_path: Path to numpy checkpoint.
        """
        # Placeholder.
        print(
            "NPZ loading not implemented. Please convert to .pth format first."
        )
        pass

    def forward_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward through encoder and collect skip connections.

        Args:
            x: Input tensor.

        Returns:
            Tuple of (encoder output, skip connections).
        """
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C

        return x, x_downsample

    def forward_up_features(
        self, x: torch.Tensor, x_downsample: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward through decoder with skip connections.

        Args:
            x: Input tensor.
            x_downsample: Skip connections from encoder.

        Returns:
            Decoded feature tensor.
        """
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[self.num_layers - 1 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C

        return x

    def up_x4(self, x: torch.Tensor) -> torch.Tensor:
        """
        Final 4x upsampling to original resolution.

        Args:
            x: Input tensor.

        Returns:
            Upsampled feature tensor.
        """
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, (
            f"input feature has wrong size, expected {H * W}, got {L}"
        )

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            x = self.output(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Segmentation logits of shape (B, num_classes, H, W)
        """
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)
        return x


def swin_unet_tiny_patch4_window7_224(
    num_classes: int = 1,
    in_chans: int = 3,
    pretrained: Optional[str] = None,
    **kwargs,
) -> SwinUNet:
    """
    Swin-UNet Tiny model with patch size 4, window size 7, image size 224.

    This is the default configuration used in the Swin-UNet paper.

    Args:
        num_classes: Number of segmentation classes
        in_chans: Number of input channels
        pretrained: Path to pretrained weights
        **kwargs: Additional arguments passed to SwinUNet

    Returns:
        SwinUNet model
    """
    model = SwinUNet(
        img_size=224,
        patch_size=4,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        depths_decoder=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        pretrained=pretrained,
        **kwargs,
    )
    return model


# Convenience aliases
SwinUNet_Tiny = swin_unet_tiny_patch4_window7_224


if __name__ == "__main__":
    # Quick test.
    model = SwinUNet(img_size=224, num_classes=4, in_chans=3)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
