#!/bin/bash

# Dataset parameters
DATASET="PH2"
TRAIN_CSV="./prepare_datasets/PH2_segs_metadata/train.csv"
VAL_CSV="./prepare_datasets/PH2_segs_metadata/val.csv"
TEST_CSV="./prepare_datasets/PH2_segs_metadata/test.csv"
UNLABELED_CSV_5K="./prepare_unsupervised_images/ISIC2020_train_subsets/train_subset_5k.csv"

# Model architecture parameters
# SEG_MODEL="unetpp" # "unetpp" or "attentionunet" or "swinunet"
# BACKBONE_ACRONYM="unetpp" # "unetpp" or "attentionunet" or "swinunet"
SEG_ENCODER="resnet34"     # Only used for U-Net variants
ENCODER_WEIGHTS="imagenet" # Only used for U-Net variants
SWIN_PRETRAINED_PATH="./checkpoints/pretrained/swin_tiny_patch4_window7_224_22k.pth"

# Seg quality predictor parameters
MQ_CHECKPOINT="./checkpoints/mq/mq_PH2/best_model.pth"

# Training hyperparameters
EPOCHS=200
BATCH_SIZE_LABELED=32
BATCH_SIZE_UNLABELED=64
LR=1e-4
WEIGHT_DECAY=1e-4
IMAGE_SIZE=224
MIXED_PRECISION="fp16"
GRAD_CLIP=1.0
EMA_DECAY=0.999

# Lambda scheduler hyperparameters
LAMBDA_INITIAL_QAR=0.0
LAMBDA_FINAL_QAR_VALUE=0.01
LAMBDA_WARMUP_QAR=30
LAMBDA_RAMPUP_QAR=40

LAMBDA_INITIAL_PL_QW=0.0
LAMBDA_FINAL_PL_QW_VALUE=0.25
LAMBDA_WARMUP_PL_QW=15
LAMBDA_RAMPUP_PL_QW=30

# Method-specific hyperparameters

# Pseudolabel (SR = sample reweighting)
CONFIDENCE_THRESHOLD_PSEUDO=0.9
CONFIDENCE_THRESHOLD_PSEUDO_SR=0.9
CONFIDENCE_TEMPERATURE_PSEUDO_SR=2.0

# Mean teacher
CONSISTENCY_TYPE="mse"

# Contrastive
LAMBDA_CONTRAST=0.1
CONFIDENCE_THRESHOLD_CONTRAST=0.9
TEMPERATURE_CONTRAST=0.1
NUM_NEGATIVES_CONTRAST=256

# ICT
ICT_ALPHA=1.0

# Ours
QUALITY_THRESHOLD_SR=0.5
QUALITY_TEMPERATURE_SR=2.0

# System hyperparameters
NUM_WORKERS=4
SEED=<SEED>

# Early stopping
PATIENCE=30

# Output and experiment tracking
OUTPUT_DIR="./checkpoints/methods/${DATASET}"
COMET_PROJECT="vqm-methods-${DATASET}"
USE_COMET=false  # Set to true to enable Comet.ml experiment tracking

# Seg models
SEG_MODELS=(
  "unetpp"
  "attentionunet"
  "swinunet"
)

# Create acronyms for seg models.
declare -A SEG_MODEL_ACRONYM=(
  [unetpp]="UP"
  [attentionunet]="AU"
  [swinunet]="SU"
)

# Create acronyms for method names.
declare -A METHOD_ACRONYM=(
  [supervised]="SUP"
  [pseudo_label]="PL-T"
  [pseudo_label_sample_reweighting]="PL-C"
  [mean_teacher]="MT"
  [ua_mt]="UA-MT"
  [mean_teacher_quality]="MT-QW"
  [ict]="ICT"
  [ict_quality]="ICT-QW"
  [cps]="CPS"
  [cps_quality_weighted]="CPS-QW"
  [contrastive]="CL"
  [contrastive_quality]="CL-QW"
  [ours_qar]="QAR"
  [ours_pl_qw]="PL-QW"
)

# Create a function.
run_method() {
  local method="$1"
  shift

  local acronym="${METHOD_ACRONYM[$method]}"
  if [[ -z "$acronym" ]]; then
    echo "Unknown method: $method" >&2
    exit 1
  fi

  local experiment_name="${acronym}_${BASE_EXPERIMENT_NAME}"

  python -u training/train_semisup.py \
    --method "$method" \
    --experiment_name "${experiment_name}" \
    "${COMMON_ARGS[@]}" \
    "${@}"
}


for SEG_MODEL in "${SEG_MODELS[@]}"; do

  SEG_ACRONYM="${SEG_MODEL_ACRONYM[$SEG_MODEL]}"
  if [[ -z "$SEG_ACRONYM" ]]; then
      echo "Unknown seg model: $SEG_MODEL" >&2
      exit 1
  fi

  # Create experiment name.
  BASE_EXPERIMENT_NAME="${DATASET}_${SEG_ACRONYM}_${EPOCHS}_${PATIENCE}_\
${BATCH_SIZE_LABELED}_${BATCH_SIZE_UNLABELED}_${LR}_${WEIGHT_DECAY}_\
${EMA_DECAY}_${SEED}"

  COMMON_ARGS=(
      --labeled_train_csv "$TRAIN_CSV"
      --labeled_val_csv "$VAL_CSV"
      --labeled_test_csv "$TEST_CSV"
      --unlabeled_csv "$UNLABELED_CSV_5K"
      --seg_model "$SEG_MODEL"
      --encoder "$SEG_ENCODER"
      --encoder_weights "$ENCODER_WEIGHTS"
      --swin_pretrained_path "$SWIN_PRETRAINED_PATH"
      --epochs "$EPOCHS"
      --batch_size_labeled "$BATCH_SIZE_LABELED"
      --batch_size_unlabeled "$BATCH_SIZE_UNLABELED"
      --lr "$LR"
      --weight_decay "$WEIGHT_DECAY"
      --image_size "$IMAGE_SIZE"
      --grad_clip "$GRAD_CLIP"
      --num_workers "$NUM_WORKERS"
      --seed "$SEED"
      --output_dir "$OUTPUT_DIR"
      --patience "$PATIENCE"
      --mixed_precision "$MIXED_PRECISION"
  )

  # Add Comet.ml arguments if enabled.
  if [ "$USE_COMET" = true ]; then
      COMMON_ARGS+=(--use_comet --comet_project "$COMET_PROJECT")
  fi

#   run_method supervised

#   run_method pseudo_label \
#       --lambda_initial "$LAMBDA_INITIAL_PL_QW" \
#       --lambda_final "$LAMBDA_FINAL_PL_QW_VALUE" \
#       --lambda_warmup "$LAMBDA_WARMUP_PL_QW" \
#       --lambda_rampup "$LAMBDA_RAMPUP_PL_QW" \
#       --confidence_threshold "$CONFIDENCE_THRESHOLD_PSEUDO"

#   run_method pseudo_label_sample_reweighting \
#       --lambda_initial "$LAMBDA_INITIAL_PL_QW" \
#       --lambda_final "$LAMBDA_FINAL_PL_QW_VALUE" \
#       --lambda_warmup "$LAMBDA_WARMUP_PL_QW" \
#       --lambda_rampup "$LAMBDA_RAMPUP_PL_QW" \
#       --confidence_threshold "$CONFIDENCE_THRESHOLD_PSEUDO_SR" \
#       --confidence_temperature "$CONFIDENCE_TEMPERATURE_PSEUDO_SR"

#   run_method mean_teacher \
#       --lambda_initial "$LAMBDA_INITIAL_PL_QW" \
#       --lambda_final "$LAMBDA_FINAL_PL_QW_VALUE" \
#       --lambda_warmup "$LAMBDA_WARMUP_PL_QW" \
#       --lambda_rampup "$LAMBDA_RAMPUP_PL_QW" \
#       --consistency_type "$CONSISTENCY_TYPE" \
#       --ema_decay "$EMA_DECAY"

#   run_method ua_mt \
#       --lambda_initial "$LAMBDA_INITIAL_PL_QW" \
#       --lambda_final "$LAMBDA_FINAL_PL_QW_VALUE" \
#       --lambda_warmup "$LAMBDA_WARMUP_PL_QW" \
#       --lambda_rampup "$LAMBDA_RAMPUP_PL_QW" \
#       --ema_decay "$EMA_DECAY"

#   run_method mean_teacher_quality \
#       --lambda_initial "$LAMBDA_INITIAL_PL_QW" \
#       --lambda_final "$LAMBDA_FINAL_PL_QW_VALUE" \
#       --lambda_warmup "$LAMBDA_WARMUP_PL_QW" \
#       --lambda_rampup "$LAMBDA_RAMPUP_PL_QW" \
#       --mq_checkpoint "$MQ_CHECKPOINT" \
#       --consistency_type "$CONSISTENCY_TYPE" \
#       --quality_threshold "$QUALITY_THRESHOLD_SR" \
#       --quality_temperature "$QUALITY_TEMPERATURE_SR"

#   run_method ict \
#       --lambda_initial "$LAMBDA_INITIAL_PL_QW" \
#       --lambda_final "$LAMBDA_FINAL_PL_QW_VALUE" \
#       --lambda_warmup "$LAMBDA_WARMUP_PL_QW" \
#       --lambda_rampup "$LAMBDA_RAMPUP_PL_QW" \
#       --ict_alpha "$ICT_ALPHA"

#   run_method ict_quality \
#       --lambda_initial "$LAMBDA_INITIAL_PL_QW" \
#       --lambda_final "$LAMBDA_FINAL_PL_QW_VALUE" \
#       --lambda_warmup "$LAMBDA_WARMUP_PL_QW" \
#       --lambda_rampup "$LAMBDA_RAMPUP_PL_QW" \
#       --mq_checkpoint "$MQ_CHECKPOINT" \
#       --ict_alpha "$ICT_ALPHA" \
#       --quality_threshold "$QUALITY_THRESHOLD_SR" \
#       --quality_temperature "$QUALITY_TEMPERATURE_SR"

#   run_method cps \
#       --lambda_initial "$LAMBDA_INITIAL_PL_QW" \
#       --lambda_final "$LAMBDA_FINAL_PL_QW_VALUE" \
#       --lambda_warmup "$LAMBDA_WARMUP_PL_QW" \
#       --lambda_rampup "$LAMBDA_RAMPUP_PL_QW" \

#   run_method contrastive \
#       --lambda_initial "$LAMBDA_INITIAL_PL_QW" \
#       --lambda_final "$LAMBDA_FINAL_PL_QW_VALUE" \
#       --lambda_warmup "$LAMBDA_WARMUP_PL_QW" \
#       --lambda_rampup "$LAMBDA_RAMPUP_PL_QW" \
#       --lambda_contrast "$LAMBDA_CONTRAST" \
#       --confidence_threshold "$CONFIDENCE_THRESHOLD_CONTRAST" \
#       --contrast_temperature "$TEMPERATURE_CONTRAST" \
#       --num_negatives "$NUM_NEGATIVES_CONTRAST"

#   run_method contrastive_quality \
#       --lambda_initial "$LAMBDA_INITIAL_PL_QW" \
#       --lambda_final "$LAMBDA_FINAL_PL_QW_VALUE" \
#       --lambda_warmup "$LAMBDA_WARMUP_PL_QW" \
#       --lambda_rampup "$LAMBDA_RAMPUP_PL_QW" \
#       --mq_checkpoint "$MQ_CHECKPOINT" \
#       --quality_threshold "$QUALITY_THRESHOLD_SR" \
#       --quality_temperature "$QUALITY_TEMPERATURE_SR" \
#       --lambda_contrast "$LAMBDA_CONTRAST" \
#       --contrast_temperature "$TEMPERATURE_CONTRAST" \
#       --num_negatives "$NUM_NEGATIVES_CONTRAST"

#   run_method cps_quality_weighted \
#       --lambda_initial "$LAMBDA_INITIAL_PL_QW" \
#       --lambda_final "$LAMBDA_FINAL_PL_QW_VALUE" \
#       --lambda_warmup "$LAMBDA_WARMUP_PL_QW" \
#       --lambda_rampup "$LAMBDA_RAMPUP_PL_QW" \
#       --mq_checkpoint "$MQ_CHECKPOINT" \
#       --quality_threshold "$QUALITY_THRESHOLD_SR" \
#       --quality_temperature "$QUALITY_TEMPERATURE_SR"

  run_method ours_qar \
      --lambda_initial "$LAMBDA_INITIAL_QAR" \
      --lambda_final "$LAMBDA_FINAL_QAR_VALUE" \
      --lambda_warmup "$LAMBDA_WARMUP_QAR" \
      --lambda_rampup "$LAMBDA_RAMPUP_QAR" \
      --mq_checkpoint "$MQ_CHECKPOINT" \
      --quality_threshold "$QUALITY_THRESHOLD_SR" \
      --quality_temperature "$QUALITY_TEMPERATURE_SR"

  run_method ours_pl_qw \
      --lambda_initial "$LAMBDA_INITIAL_PL_QW" \
      --lambda_final "$LAMBDA_FINAL_PL_QW_VALUE" \
      --lambda_warmup "$LAMBDA_WARMUP_PL_QW" \
      --lambda_rampup "$LAMBDA_RAMPUP_PL_QW" \
      --mq_checkpoint "$MQ_CHECKPOINT" \
      --quality_threshold "$QUALITY_THRESHOLD_SR" \
      --quality_temperature "$QUALITY_TEMPERATURE_SR"

done
