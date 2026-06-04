"""
Baselines for semi-supervised segmentation.

Contains:
- supervised: Supervised-only training (baseline)
- pseudo_labeling: Pseudo-labeling baseline
"""

from .cl import (
    train_contrastive_pseudo_label,
    train_contrastive_pseudo_label_quality,
)
from .cps import train_cps, train_cps_quality_weighted
from .ict import train_ict, train_ict_quality
from .mt import train_mean_teacher, train_mean_teacher_quality, train_ua_mt
from .ours import train_ours_pl_qw, train_ours_qar
from .pseudolabels import (
    train_pseudo_label,
    train_pseudo_label_sample_reweighting,
)
from .supervised import train_supervised

__all__ = [
    # Supervised only.
    "train_supervised",
    # Pseudo-labeling.
    "train_pseudo_label",
    "train_pseudo_label_sample_reweighting",
    # Mean Teacher.
    "train_mean_teacher",
    "train_ua_mt",
    "train_mean_teacher_quality",
    # Interpolation Consistency Training.
    "train_ict",
    "train_ict_quality",
    # Cross Pseudo Supervision.
    "train_cps",
    "train_cps_quality_weighted",
    # Contrastive Pseudo Labeling.
    "train_contrastive_pseudo_label",
    "train_contrastive_pseudo_label_quality",
    # Ours.
    "train_ours_qar",
    "train_ours_pl_qw",
]
