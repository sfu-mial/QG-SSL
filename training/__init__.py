"""
Training module.

Contains:
- losses: Segmentation and quality-guided losses
- schedulers: Lambda scheduling and quality monitoring
- train_mq: Training script for g_φ
- train_semisup: Semi-supervised training script
"""

from .eval_utils import evaluate, evaluate_cps, validate, validate_cps
from .losses import (
    BCEDiceLoss,
    DiceLoss,
    FocalLoss,
    PerSampleBCEDiceLoss,
    QualityGuidedLoss,
)
from .methods import (
    train_contrastive_pseudo_label,
    train_contrastive_pseudo_label_quality,
    train_cps,
    train_cps_quality_weighted,
    train_ict,
    train_ict_quality,
    train_mean_teacher,
    train_mean_teacher_quality,
    train_ours_pl_qw,
    train_ours_qar,
    train_pseudo_label,
    train_pseudo_label_sample_reweighting,
    train_supervised,
    train_ua_mt,
)
from .metrics import (
    dice_coefficient,
    f1_score,
    f1_score_batch_level,
    jaccard_coefficient,
    pixel_accuracy,
)
from .schedulers import LambdaScheduler, LambdaSchedulerConfig

__all__ = [
    # Evaluation
    "evaluate",
    "evaluate_cps",
    "validate",
    "validate_cps",
    # Losses
    "DiceLoss",
    "BCEDiceLoss",
    "FocalLoss",
    "PerSampleBCEDiceLoss",
    "QualityGuidedLoss",
    # Metrics
    "dice_coefficient",
    "jaccard_coefficient",
    "f1_score",
    "f1_score_batch_level",
    "pixel_accuracy",
    # Schedulers
    "LambdaScheduler",
    "LambdaSchedulerConfig",
    # Methods
    "train_supervised",
    "train_pseudo_label",
    "train_pseudo_label_sample_reweighting",
    "train_mean_teacher",
    "train_ua_mt",
    "train_mean_teacher_quality",
    "train_ict",
    "train_ict_quality",
    "train_cps",
    "train_cps_quality_weighted",
    "train_contrastive_pseudo_label",
    "train_contrastive_pseudo_label_quality",
    "train_ours_qar",
    "train_ours_pl_qw",
]
