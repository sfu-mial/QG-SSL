"""
Data module for Segmentation Quality Predictor training.

Components:
- corruption_ops: Individual corruption functions.
- vqm_generator: On-the-fly VQM generation with corruption sampling.
- mq_dataset: PyTorch Mask Quality (MQ) datasets and dataloaders for g_φ training.
"""

from .corruption_ops import (  # Morphological; Shape approximations; Boundary perturbations; Structural; Geometric; Blob approximations; Catastrophic; Identity; Utilities
    CATASTROPHIC_CORRUPTION_REGISTRY,
    CORRUPTION_REGISTRY,
    HEAVY_CORRUPTION_REGISTRY,
    add_holes,
    add_islands,
    apply_corruption,
    bounding_box_fill,
    circular_blob,
    closing,
    convex_hull,
    dilation,
    drop_random_component,
    elastic_deformation,
    ellipse_approximation,
    empty_mask,
    erosion,
    full_mask,
    gaussian_blob,
    get_all_corruption_names,
    grid_distortion,
    holes_and_islands,
    identity,
    invert_mask,
    jagged_boundary,
    keep_largest_component,
    opening,
    polygon_approximation,
    random_affine,
    random_mask,
    random_morphology,
    random_rectangle_dropout,
    rotate_mask,
    scale_mask,
    shift_mask,
    smooth_boundary,
)
from .mq_dataset import (
    MQDataset,
    MQDatasetWithWeakModels,
    get_mq_dataloaders,
    get_mq_dataloaders_with_weak_models,
    get_spatial_transforms,
)
from .vqm_generator import CorruptionConfig, VQMGenerator, dice_coefficient
from .weak_model_corruption import (
    SimpleSegDataset,
    WeakModelCorruptor,
    create_extended_vqm_generator,
    train_weak_models,
)

__all__ = [
    # Corruption ops
    "erosion",
    "dilation",
    "random_morphology",
    "opening",
    "closing",
    "ellipse_approximation",
    "polygon_approximation",
    "convex_hull",
    "bounding_box_fill",
    "jagged_boundary",
    "smooth_boundary",
    "elastic_deformation",
    "grid_distortion",
    "add_holes",
    "add_islands",
    "holes_and_islands",
    "keep_largest_component",
    "drop_random_component",
    "random_rectangle_dropout",
    "shift_mask",
    "scale_mask",
    "rotate_mask",
    "random_affine",
    "gaussian_blob",
    "circular_blob",
    "invert_mask",
    "empty_mask",
    "full_mask",
    "random_mask",
    "identity",
    "apply_corruption",
    "get_all_corruption_names",
    "CORRUPTION_REGISTRY",
    "HEAVY_CORRUPTION_REGISTRY",
    "CATASTROPHIC_CORRUPTION_REGISTRY",
    # VQM generator
    "VQMGenerator",
    "CorruptionConfig",
    "dice_coefficient",
    # Dataset
    "MQDataset",
    "MQDatasetWithWeakModels",
    "get_mq_dataloaders",
    "get_mq_dataloaders_with_weak_models",
    "get_spatial_transforms",
    # Weak model corruption
    "WeakModelCorruptor",
    "SimpleSegDataset",
    "train_weak_models",
    "create_extended_vqm_generator",
]
