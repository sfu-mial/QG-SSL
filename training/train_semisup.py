"""
Semi-Supervised Segmentation Training with HuggingFace Accelerate.

Supports multiple methods:
- supervised: Supervised only (baseline)
- pseudo_label: Pseudo-labeling
- pseudo_label_sample_reweighting: Pseudo-labeling with sample reweighting
- mean_teacher: Mean Teacher
- ua_mt: Uncertainty-Aware Mean Teacher
- cps: Cross Pseudo Supervision
- contrastive: Contrastive learning with pseudo-labels
- contrastive_quality: Contrastive learning with quality-weighted pseudo-labels
- ours_qar: Quality-guided loss (our method)
- ours_pl_qw: Quality-guided sample reweighting (our method)

Usage:
    accelerate launch train_semisup.py \
        --method ours_qar \
        --labeled_train_csv /path/to/train.csv \
        --labeled_val_csv /path/to/val.csv \
        --unlabeled_csv /path/to/unlabeled.csv \
        --mq_checkpoint /path/to/mq_best_model.pth \
        --output_dir ./checkpoints/semisup
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from accelerate import Accelerator
from loguru import logger
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.seg_datasets import LabeledSegmentationDataset, UnlabeledDataset
from models import SwinUNet, load_pretrained_swin_unet
from models.ema import create_ema_model
from models.quality_predictor import create_quality_predictor
from training.eval_utils import evaluate, evaluate_cps, validate, validate_cps
from training.methods import (
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
from training.schedulers import LambdaScheduler, LambdaSchedulerConfig

try:
    from comet_ml import Experiment

    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    Experiment = None


# =============================================================================
# MODEL FACTORY
# =============================================================================


def create_segmentation_model(
    model_name: str,
    encoder: str = "resnet34",  # Only used for U-Net++ and Attention U-Net.
    encoder_weights: str = "imagenet",  # Only used for U-Net++ and Attention U-Net.
    swin_pretrained_path: Optional[str] = None,
):
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            "Please install: pip install segmentation-models-pytorch"
        )

    models = {
        "unet": smp.Unet,
        "unetpp": smp.UnetPlusPlus,
        "deeplabv3": smp.DeepLabV3,
        "deeplabv3p": smp.DeepLabV3Plus,
        "fpn": smp.FPN,
        "pspnet": smp.PSPNet,
        "manet": smp.MAnet,
        "attentionunet": smp.AttentionUnet,
        "swinunet": SwinUNet,
    }

    if model_name not in models:
        print(f"Unknown model {model_name}, using U-Net++")
        model_name = "unetpp"

    if model_name == "swinunet":
        model = SwinUNet(img_size=224, num_classes=1, in_chans=3)
        load_pretrained_swin_unet(model, pretrained_path=swin_pretrained_path)
        return model
    else:
        return models[model_name](
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def set_seed(seed: int):
    """
    Set the random seed for reproducibility.

    Args:
        seed: Seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class EarlyStopper:
    """
    Early stopping helper.

    Args:
        patience: Number of epochs to wait before stopping.
        mode: 'max' for maximizing, 'min' for minimizing.
    """

    def __init__(self, patience: int, mode: str = "max"):
        """
        Initialize the early stopper.
        """
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best = float("-inf") if mode == "max" else float("inf")

    def should_stop(self, value: float) -> Tuple[bool, bool]:
        """
        Check if training should stop.

        Args:
            value: Current metric value.

        Returns:
            Tuple of (should_stop, improved).
        """
        improved = (
            (value > self.best) if self.mode == "max" else (value < self.best)
        )
        if improved:
            self.best = value
            self.counter = 0
            return False, True
        self.counter += 1
        return self.counter >= self.patience, False


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        ArgumentParser object.
    """
    parser = argparse.ArgumentParser(
        description="Semi-Supervised Segmentation Training"
    )

    # Method selection.
    parser.add_argument(
        "--method",
        type=str,
        default="ours_qar",
        choices=[
            "supervised",
            "pseudo_label",
            "pseudo_label_sample_reweighting",
            "mean_teacher",
            "mean_teacher_quality",
            "ua_mt",
            "ict",
            "ict_quality",
            "cps",
            "cps_quality_weighted",
            "contrastive",
            "contrastive_quality",
            "ours_qar",
            "ours_pl_qw",
        ],
        help="Training method",
    )

    # Data paths.
    parser.add_argument("--labeled_train_csv", type=str, required=True)
    parser.add_argument("--labeled_val_csv", type=str, required=True)
    parser.add_argument("--labeled_test_csv", type=str, default=None)
    parser.add_argument(
        "--unlabeled_csv",
        type=str,
        default=None,
        help="Required for all methods except 'supervised'",
    )

    # Model architectures.
    parser.add_argument("--seg_model", type=str, default="unetpp")
    parser.add_argument("--seg_encoder", type=str, default="resnet34")
    parser.add_argument(
        "--encoder_weights",
        type=str,
        default="imagenet",
        help="'imagenet' or 'None' for random init",
    )
    parser.add_argument(
        "--swin_pretrained_path",
        type=str,
        default=None,
        help="Required for using SwinUNet",
    )
    parser.add_argument(
        "--mq_checkpoint",
        type=str,
        default=None,
        help="Required for method='ours_qar'",
    )

    # Training hyper-parameters.
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size_labeled", type=int, default=8)
    parser.add_argument("--batch_size_unlabeled", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Lambda scheduling for ramp-up.
    parser.add_argument("--lambda_initial", type=float, default=0.0)
    parser.add_argument("--lambda_final", type=float, default=0.5)
    parser.add_argument("--lambda_rampup", type=int, default=20)
    parser.add_argument("--lambda_warmup", type=int, default=10)

    # Method-specific hyper-parameters.
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.9,
        help="For pseudo_label, pseudo_label_sample_reweighting, \
        and contrastive methods",
    )
    parser.add_argument(
        "--confidence_temperature",
        type=float,
        default=2.0,
        help="For pseudo_label_sample_reweighting method",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="For both MT and both ours_qar methods",
    )
    parser.add_argument(
        "--consistency_type",
        type=str,
        default="mse",
        choices=["mse", "kl"],
        help="For mean_teacher",
    )

    # ICT-specific hyper-parameters.
    parser.add_argument(
        "--ict_alpha",
        type=float,
        default=1.0,
        help="α for ICT's Beta(α, α) distribution",
    )

    # Contrastive learning-specific hyper-parameters.
    parser.add_argument(
        "--lambda_contrast",
        type=float,
        default=0.1,
        help="Weight for contrastive loss (for contrastive method)",
    )
    parser.add_argument(
        "--contrast_temperature",
        type=float,
        default=0.1,
        help="Temperature for contrastive loss (for contrastive method)",
    )
    parser.add_argument(
        "--num_negatives",
        type=int,
        default=256,
        help="Number of negative samples for contrastive loss",
    )

    # Quality reweighting-specific hyper-parameters.
    parser.add_argument(
        "--quality_threshold",
        type=float,
        default=0.5,
        help="For ours_pl_qw method",
    )
    parser.add_argument(
        "--quality_temperature",
        type=float,
        default=2.0,
        help="For ours_pl_qw method",
    )

    # System hyper-parameters.
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints/semisup"
    )
    parser.add_argument("--patience", type=int, default=30)

    # 🤗 Hugging Face Accelerate AMP settings.
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
    )

    # Comet logging setup.
    parser.add_argument(
        "--use_comet", action="store_true", help="Enable Comet ML logging"
    )
    parser.add_argument("--comet_project", type=str, default="vqm-methods")
    parser.add_argument("--experiment_name", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Initialize accelerator object.
    # This includes specifying that the logger will be Comet.
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        rng_types=["torch", "cuda", "generator"],
        log_with="comet_ml" if args.use_comet else None,
    )

    # Validate arguments.
    if args.method != "supervised" and args.unlabeled_csv is None:
        raise ValueError(
            f"--unlabeled_csv required for method '{args.method}'"
        )
    if (
        args.method
        in [
            "ours_qar",
            "ours_pl_qw",
            "cps_quality_weighted",
            "ict_quality",
            "mean_teacher_quality",
            "contrastive_quality",
        ]
        and args.mq_checkpoint is None
    ):
        raise ValueError(
            "--mq_checkpoint required for methods using quality predictor: "
            "['ours_qar', 'ours_pl_qw', 'cps_quality_weighted', "
            "'ict_quality', 'mean_teacher_quality', or 'contrastive_quality'"
        )

    # Create the output directory (only on the main process).
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.experiment_name is None:
        args.experiment_name = f"{args.method}_{timestamp}"
    output_dir = Path(args.output_dir) / args.experiment_name
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Method: {args.method}")
        print(f"Output: {output_dir}")
        print(f"Mixed precision: {args.mixed_precision}")

        with open(output_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

        # Remove the default logging handler (terminal output).
        logger.remove()

        # Add a handler for terminal output logging but without timestamps.
        logger.add(sys.stderr, format="{message}", level="INFO")

        # Add a handler for the output log file with timestamps.
        logger.add(
            output_dir / "train.log",
            format="{time:YYYY-MM-DD at HH:mm:ss} | {message}",
            level="INFO",
        )

        """
        # Create tee output class to log to both terminal and file.
        class TeeOutput:
            def __init__(self, original):
                self.original = original

            def write(self, msg):
                # Write to terminal as usual.
                self.original.write(msg)
                if msg.strip():
                    # Write to file with timestamp using loguru.
                    logger.info(msg.strip())

            def flush(self):
                self.original.flush()

        # Redirect stdout to tee output.
        sys.stdout = TeeOutput(sys.stdout)
        """

    accelerator.wait_for_everyone()

    # Initialize Comet experiment (main process only).
    experiment = None
    if args.use_comet and accelerator.is_main_process:
        if COMET_AVAILABLE:
            accelerator.init_trackers(
                project_name=args.comet_project, config=vars(args)
            )
            experiment = accelerator.get_tracker("comet_ml").tracker
            # experiment = Experiment(
            #     project_name=args.comet_project,
            #     auto_metric_logging=False,
            #     auto_param_logging=False,
            # )
            experiment.set_name(args.experiment_name)
            experiment.log_parameters(vars(args))
            print(f"  Comet experiment: {experiment.get_key()}")
        else:
            print("  Warning: comet_ml not installed, skipping Comet logging")

    # Create dataset objects using the helper functions.
    if accelerator.is_main_process:
        print("\nLoading datasets...")
    train_labeled = LabeledSegmentationDataset(
        args.labeled_train_csv, args.image_size, is_train=True
    )
    val_labeled = LabeledSegmentationDataset(
        args.labeled_val_csv, args.image_size, is_train=False
    )
    test_labeled = LabeledSegmentationDataset(
        args.labeled_test_csv, args.image_size, is_train=False
    )
    if accelerator.is_main_process:
        print(
            f"Labeled train: {len(train_labeled)}\n"
            f"        val: {len(val_labeled)}\n"
            f"        test: {len(test_labeled)}"
        )

    # Create PyTorch DataLoader objects for the labeled datasets.
    labeled_loader = DataLoader(
        train_labeled,
        args.batch_size_labeled,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_labeled,
        args.batch_size_labeled,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_labeled,
        args.batch_size_labeled,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # Create PyTorch DataLoader object for the unlabeled dataset.
    unlabeled_loader = None
    if args.method != "supervised":
        train_unlabeled = UnlabeledDataset(args.unlabeled_csv, args.image_size)
        if accelerator.is_main_process:
            print(f"  Unlabeled: {len(train_unlabeled)}")
        unlabeled_loader = DataLoader(
            train_unlabeled,
            args.batch_size_unlabeled,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    # Create the segmentation model(s) f_θ.
    encoder_weights = (
        None
        if args.encoder_weights.lower() == "none"
        else args.encoder_weights
    )
    if accelerator.is_main_process:
        print(
            f"\nCreating {args.seg_model} with {args.seg_encoder} encoder (weights={encoder_weights})..."
        )

    # For CPS and its variant (quality-weighted CPS), we need two models with
    # different initializations.
    if args.method in ["cps", "cps_quality_weighted"]:
        # Create first model with current seed.
        seg_model1 = create_segmentation_model(
            args.seg_model, args.seg_encoder, encoder_weights
        )
        # Create second model with different seed for different initialization.
        # We just add an offset to the seed to make it different from the current seed.
        set_seed(args.seed + 1000)
        seg_model2 = create_segmentation_model(
            args.seg_model, args.seg_encoder, encoder_weights
        )
        set_seed(args.seed)  # Reset seed.

        if accelerator.is_main_process:
            print(
                f"  Parameters (per model): {sum(p.numel() for p in seg_model1.parameters()):,}"
            )
    # For all other methods, we only need one model.
    else:
        seg_model = create_segmentation_model(
            args.seg_model, args.seg_encoder, encoder_weights
        )
        if accelerator.is_main_process:
            print(
                f"  Parameters: {sum(p.numel() for p in seg_model.parameters()):,}"
            )

    # Quality predictor (for 'ours_qar' method and variants) - **not** prepared by
    # accelerator; stays on device manually.
    quality_predictor = None
    if args.method in [
        "ours_qar",
        "ours_pl_qw",
        "cps_quality_weighted",
        "ict_quality",
        "mean_teacher_quality",
        "contrastive_quality",
    ]:
        if accelerator.is_main_process:
            print(f"\nLoading g_φ from {args.mq_checkpoint}...")
        checkpoint = torch.load(
            args.mq_checkpoint,
            map_location=accelerator.device,
            # weights_only=False,
        )
        mq_args = checkpoint.get("args", {})
        quality_predictor = create_quality_predictor(
            backbone=mq_args.get("backbone", "resnet18.a1_in1k"),
            pretrained=False,
        )
        quality_predictor.load_state_dict(checkpoint)
        quality_predictor = quality_predictor.to(accelerator.device).eval()
        for p in quality_predictor.parameters():
            p.requires_grad = False
        if accelerator.is_main_process:
            print(
                f"  Loaded (val_corr: {checkpoint.get('val_correlation', 'N/A')})"
            )

    # Optimizer(s).
    # For CPS and its variant (quality-weighted CPS), we need two optimizers,
    # one for each model.
    if args.method in ["cps", "cps_quality_weighted"]:
        optimizer1 = optim.AdamW(
            seg_model1.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        optimizer2 = optim.AdamW(
            seg_model2.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        lr_scheduler1 = optim.lr_scheduler.CosineAnnealingLR(
            optimizer1, T_max=args.epochs, eta_min=1e-6
        )
        lr_scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
            optimizer2, T_max=args.epochs, eta_min=1e-6
        )
    # For all other methods, only one optimizer.
    else:
        optimizer = optim.AdamW(
            seg_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=10, T_mult=2, eta_min=1e-6
    # )

    # "Prepare" with accelerator.
    # This means "preparing" the seg_model f_θ, the optimizer(s), the data loaders,
    # and the LR scheduler(s).
    if args.method == "supervised":
        (
            seg_model,
            optimizer,
            labeled_loader,
            val_loader,
            test_loader,
            lr_scheduler,
        ) = accelerator.prepare(
            seg_model,
            optimizer,
            labeled_loader,
            val_loader,
            test_loader,
            lr_scheduler,
        )
    # Again, since CPS needs two models, we need to prepare them both.
    elif args.method in ["cps", "cps_quality_weighted"]:
        (
            seg_model1,
            seg_model2,
            optimizer1,
            optimizer2,
            labeled_loader,
            unlabeled_loader,
            val_loader,
            test_loader,
            lr_scheduler1,
            lr_scheduler2,
        ) = accelerator.prepare(
            seg_model1,
            seg_model2,
            optimizer1,
            optimizer2,
            labeled_loader,
            unlabeled_loader,
            val_loader,
            test_loader,
            lr_scheduler1,
            lr_scheduler2,
        )
    # For all other methods, we only need one model.
    else:
        (
            seg_model,
            optimizer,
            labeled_loader,
            unlabeled_loader,
            val_loader,
            test_loader,
            lr_scheduler,
        ) = accelerator.prepare(
            seg_model,
            optimizer,
            labeled_loader,
            unlabeled_loader,
            val_loader,
            test_loader,
            lr_scheduler,
        )

    # EMA model (created **after** accelerator.prepare, uses unwrapped model).
    ema_model = None
    if args.method in [
        "mean_teacher",
        "ua_mt",
        "mean_teacher_quality",
        "ict",
        "ict_quality",
        "contrastive_quality",
        "ours_qar",
        "ours_pl_qw",
    ]:
        unwrapped_model = accelerator.unwrap_model(seg_model)
        ema_model = create_ema_model(
            unwrapped_model,
            args.ema_decay,
            use_rampup=True,
            device=accelerator.device,
        )
        if accelerator.is_main_process:
            print(f"  EMA enabled (decay={args.ema_decay})")

    # Lambda scheduler.
    lambda_scheduler = LambdaScheduler(
        LambdaSchedulerConfig(
            initial_lambda=args.lambda_initial,
            final_lambda=args.lambda_final,
            rampup_epochs=args.lambda_rampup,
            warmup_epochs=args.lambda_warmup,
        )
    )

    # Training loop.
    # Initialize early stopper. Set best_dice to 0.0.
    early_stopper = EarlyStopper(args.patience, mode="max")
    best_dice = 0.0

    if accelerator.is_main_process:
        print(f"\nStarting {args.method} training...")

    for epoch in range(args.epochs):
        current_lambda = (
            lambda_scheduler.step(epoch)
            if args.method != "supervised"
            else 0.0
        )

        # Select training function based on method.
        if args.method == "supervised":
            train_metrics = train_supervised(
                seg_model,
                labeled_loader,
                optimizer,
                accelerator,
                epoch,
                args.grad_clip,
            )
        elif args.method == "pseudo_label":
            train_metrics = train_pseudo_label(
                seg_model,
                labeled_loader,
                unlabeled_loader,
                optimizer,
                current_lambda,
                accelerator,
                epoch,
                args.confidence_threshold,
                args.grad_clip,
            )
        elif args.method == "pseudo_label_sample_reweighting":
            train_metrics = train_pseudo_label_sample_reweighting(
                seg_model,
                labeled_loader,
                unlabeled_loader,
                optimizer,
                current_lambda,
                accelerator,
                epoch,
                args.grad_clip,
                args.confidence_threshold,
                args.confidence_temperature,
            )
        elif args.method == "mean_teacher":
            train_metrics = train_mean_teacher(
                seg_model,
                ema_model,
                labeled_loader,
                unlabeled_loader,
                optimizer,
                current_lambda,
                accelerator,
                epoch,
                args.consistency_type,
                args.grad_clip,
            )
        elif args.method == "ua_mt":
            train_metrics = train_ua_mt(
                seg_model,
                ema_model,
                labeled_loader,
                unlabeled_loader,
                optimizer,
                current_lambda,
                accelerator,
                epoch,
                args.grad_clip,
            )
        elif args.method == "mean_teacher_quality":
            train_metrics = train_mean_teacher_quality(
                seg_model,
                ema_model,
                quality_predictor,
                labeled_loader,
                unlabeled_loader,
                optimizer,
                current_lambda,
                accelerator,
                epoch,
                args.consistency_type,
                args.grad_clip,
                args.quality_threshold,
                args.quality_temperature,
            )
        elif args.method == "ict":
            train_metrics = train_ict(
                seg_model,
                ema_model,
                labeled_loader,
                unlabeled_loader,
                optimizer,
                current_lambda,
                accelerator,
                epoch,
                args.grad_clip,
                args.ict_alpha,
            )
        elif args.method == "ict_quality":
            train_metrics = train_ict_quality(
                seg_model,
                ema_model,
                quality_predictor,
                labeled_loader,
                unlabeled_loader,
                optimizer,
                current_lambda,
                accelerator,
                epoch,
                args.grad_clip,
                args.ict_alpha,
                args.quality_threshold,
                args.quality_temperature,
            )
        elif args.method == "cps":
            train_metrics = train_cps(
                seg_model1,
                seg_model2,
                labeled_loader,
                unlabeled_loader,
                optimizer1,
                optimizer2,
                current_lambda,
                accelerator,
                epoch,
                args.grad_clip,
            )
        elif args.method == "cps_quality_weighted":
            train_metrics = train_cps_quality_weighted(
                seg_model1,
                seg_model2,
                quality_predictor,
                labeled_loader,
                unlabeled_loader,
                optimizer1,
                optimizer2,
                current_lambda,
                accelerator,
                epoch,
                args.grad_clip,
                args.quality_threshold,
                args.quality_temperature,
            )
        elif args.method == "contrastive":
            train_metrics = train_contrastive_pseudo_label(
                seg_model,
                labeled_loader,
                unlabeled_loader,
                optimizer,
                current_lambda,
                args.lambda_contrast,
                accelerator,
                epoch,
                args.grad_clip,
                args.confidence_threshold,
                args.contrast_temperature,
                args.num_negatives,
            )
        elif args.method == "contrastive_quality":
            train_metrics = train_contrastive_pseudo_label_quality(
                seg_model,
                quality_predictor,
                labeled_loader,
                unlabeled_loader,
                optimizer,
                current_lambda,
                args.lambda_contrast,
                accelerator,
                epoch,
                args.grad_clip,
                args.contrast_temperature,
                args.num_negatives,
                args.quality_threshold,
                args.quality_temperature,
                ema_model,
            )
        elif args.method == "ours_qar":
            train_metrics = train_ours_qar(
                seg_model,
                quality_predictor,
                labeled_loader,
                unlabeled_loader,
                optimizer,
                current_lambda,
                accelerator,
                epoch,
                args.grad_clip,
                ema_model,
            )
        elif args.method == "ours_pl_qw":
            train_metrics = train_ours_pl_qw(
                seg_model,
                quality_predictor,
                labeled_loader,
                unlabeled_loader,
                optimizer,
                current_lambda,
                accelerator,
                epoch,
                args.grad_clip,
                ema_model,
                args.quality_threshold,
                args.quality_temperature,
            )

        # Validation.
        if args.method in ["cps", "cps_quality_weighted"]:
            val_metrics = validate_cps(
                seg_model1, seg_model2, val_loader, accelerator, epoch
            )
            lr_scheduler1.step(epoch)
            lr_scheduler2.step(epoch)
            current_lr = optimizer1.param_groups[0]["lr"]
        else:
            val_metrics = validate(
                seg_model, val_loader, accelerator, epoch, quality_predictor
            )
            lr_scheduler.step(epoch)
            current_lr = optimizer.param_groups[0]["lr"]

        # Logging (main process only).
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch + 1}/{args.epochs}:")
            train_str = ", ".join(
                [f"{k}: {v:.4f}" for k, v in train_metrics.items()]
            )
            print(f"  Train - {train_str}")
            val_str = ", ".join(
                [f"{k}: {v:.4f}" for k, v in val_metrics.items()]
            )
            print(f"  Val   - {val_str}")
            print(f"  λ={current_lambda:.3f}, LR={current_lr:.2e}")

            # Comet logging.
            if experiment is not None:
                for k, v in train_metrics.items():
                    experiment.log_metric(f"train_{k}", v, step=epoch)
                for k, v in val_metrics.items():
                    experiment.log_metric(k, v, step=epoch)
                experiment.log_metric("lambda", current_lambda, step=epoch)
                experiment.log_metric("lr", current_lr, step=epoch)

        # Save model.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if args.method in ["cps", "cps_quality_weighted"]:
                unwrapped_model1 = accelerator.unwrap_model(seg_model1)
                unwrapped_model2 = accelerator.unwrap_model(seg_model2)
                accelerator.save(
                    {
                        "model1": unwrapped_model1.state_dict(),
                        "model2": unwrapped_model2.state_dict(),
                    },
                    output_dir / f"model_{epoch:03d}.pth",
                )
            else:
                unwrapped_model = accelerator.unwrap_model(seg_model)
                accelerator.save(
                    unwrapped_model.state_dict(),
                    output_dir / f"model_{epoch:03d}.pth",
                )

        should_stop, is_best = early_stopper.should_stop(
            val_metrics["val_dice"]
        )

        if is_best:
            best_dice = val_metrics["val_dice"]

            # Save model checkpoint (main process only).
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                if args.method in ["cps", "cps_quality_weighted"]:
                    unwrapped_model1 = accelerator.unwrap_model(seg_model1)
                    unwrapped_model2 = accelerator.unwrap_model(seg_model2)
                    accelerator.save(
                        {
                            "model1": unwrapped_model1.state_dict(),
                            "model2": unwrapped_model2.state_dict(),
                        },
                        output_dir / "best_model.pth",
                    )
                else:
                    unwrapped_model = accelerator.unwrap_model(seg_model)
                    accelerator.save(
                        unwrapped_model.state_dict(),
                        output_dir / "best_model.pth",
                    )

            print(f"   New best model (Dice: {best_dice:.4f})")
            print(f"  Saved to {output_dir / 'best_model.pth'}")

        if should_stop:
            if accelerator.is_main_process:
                print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Test evaluation.
    if args.labeled_test_csv:
        if accelerator.is_main_process:
            print("\nEvaluating on test set...")

        # Load best model.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(seg_model)
            unwrapped_model.load_state_dict(
                torch.load(
                    output_dir / "best_model.pth",
                    map_location=accelerator.device,
                )
            )

        accelerator.wait_for_everyone()

        if args.method in ["cps", "cps_quality_weighted"]:
            test_metrics = evaluate_cps(
                seg_model1, seg_model2, test_loader, accelerator, epoch
            )
        else:
            test_metrics = evaluate(
                seg_model, test_loader, accelerator, quality_predictor
            )

        if accelerator.is_main_process:
            print(
                f"  Test  - Dice: {test_metrics['dice']['mean']:.4f} ± {test_metrics['dice']['std']:.4f}"
            )

            # Comet logging for test.
            if experiment is not None:
                for metric_name, metric_value in test_metrics.items():
                    experiment.log_metric(
                        f"test_{metric_name}_mean",
                        metric_value["mean"],
                        step=epoch,
                    )
                    experiment.log_metric(
                        f"test_{metric_name}_std",
                        metric_value["std"],
                        step=epoch,
                    )

            with open(output_dir / "test_results.json", "w") as f:
                json.dump(test_metrics, f, indent=2)

    if accelerator.is_main_process:
        print("\nTraining complete!")
        print(f"Method: {args.method}")
        print(f"Best Val Dice: {best_dice:.4f}")
        print(f"Saved to: {output_dir}")

        # End Comet experiment.
        if experiment is not None:
            experiment.log_metric("best_val_dice", best_dice)
            experiment.end()


if __name__ == "__main__":
    main()
