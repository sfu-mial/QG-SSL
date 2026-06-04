#!/bin/bash

# Dataset parameters
DATASET="PH2"
TRAIN_CSV="./prepare_datasets/PH2_segs_metadata/train.csv"
VAL_CSV="./prepare_datasets/PH2_segs_metadata/val.csv"
TEST_CSV="./prepare_datasets/PH2_segs_metadata/test.csv"

# Model architecture parameters
BACKBONE="resnet18.a1_in1k"
BACKBONE_ACRONYM="rn18"
DROPOUT=0.15

# Training hyperparameters
EPOCHS=150
BATCH_SIZE=32
LR=3e-4
WEIGHT_DECAY=5e-4
IMAGE_SIZE=224
SAMPLES_PER_IMAGE=50 # 140 training image x 50 samples per image = 7000 samples per epoch
SAMPLES_PER_IMAGE_VAL=5
LOSS_FUNC="smoothl1" # "mse" or "smoothl1"
SCHEDULER="cosine" # "cosine" or "onecycle"

# Augmentation (corruption) hyperparameters
IDENTITY_PROB=0.20
CROSS_IMAGE_PROB=0.10

# Weak model corruption hyperparameters
USE_WEAK_MODELS=true
WEAK_MODEL_PROB=0.05

# System hyperparameters
NUM_WORKERS=0
# DEVICE="cuda"
SEED=<SEED>

# Early stopping
PATIENCE=25

# Output and experiment tracking
OUTPUT_DIR="./checkpoints/mq"
USE_COMET=false  # Set to true to enable Comet.ml experiment tracking


# =============================================================================
# STEP 1: Train weak models
# =============================================================================

WEAK_CHECKPOINT_DIR="./checkpoints/weak_models/${DATASET}"

if [ "$USE_WEAK_MODELS" = true ] && [ ! -d "$WEAK_CHECKPOINT_DIR" ]; then
    echo "======================================================="
    echo "Training weak models for corruption diversity"
    echo "======================================================="

    python data/weak_model_corruption.py \
        --train_csv $TRAIN_CSV \
        --output_dir $WEAK_CHECKPOINT_DIR \
        --save_epochs 1 3 5 10 15 20 \
        --max_epochs 20 \
        --encoder_name resnet18 \
        --batch_size 16 \
        --lr 1e-3 \
        --image_size $IMAGE_SIZE \
        --seed $SEED
    echo ""
    echo "Weak models trained and saved to $WEAK_CHECKPOINT_DIR"
    echo "======================================================="
    echo ""
fi


# =============================================================================
# STEP 2: Train the quality predictor (g_φ)
# =============================================================================

if [ "$USE_WEAK_MODELS" = true ]; then
    WEAK_ARGS="--weak_checkpoint_paths \
    ${WEAK_CHECKPOINT_DIR}/weak_model_epoch_1.pth \
    ${WEAK_CHECKPOINT_DIR}/weak_model_epoch_3.pth \
    ${WEAK_CHECKPOINT_DIR}/weak_model_epoch_5.pth \
    ${WEAK_CHECKPOINT_DIR}/weak_model_epoch_10.pth \
    ${WEAK_CHECKPOINT_DIR}/weak_model_epoch_15.pth \
    ${WEAK_CHECKPOINT_DIR}/weak_model_epoch_20.pth \
    --weak_model_prob $WEAK_MODEL_PROB"
    EXPERIMENT_SUFFIX="_weakmodel_${WEAK_MODEL_PROB}"
else
    WEAK_ARGS=""
    EXPERIMENT_SUFFIX=""
fi

EXPERIMENT_NAME="mq_${DATASET}_${BACKBONE_ACRONYM}_${DROPOUT}_${EPOCHS}_${BATCH_SIZE}_${LR}_\
${WEIGHT_DECAY}_${SAMPLES_PER_IMAGE}_${SAMPLES_PER_IMAGE_VAL}_${IDENTITY_PROB}_\
${CROSS_IMAGE_PROB}_${LOSS_FUNC}_${SCHEDULER}_${SEED}${EXPERIMENT_SUFFIX}"

COMET_ARGS=()
if [ "$USE_COMET" = true ]; then
    COMET_ARGS=(--use_comet --comet_project seg-quality-predictor)
fi

echo "======================================================="
echo "Training g_φ (Segmentation Quality Predictor)"
echo "======================================================="

python training/train_mq.py \
    --train_csv $TRAIN_CSV \
    --val_csv $VAL_CSV \
    --test_csv $TEST_CSV \
    --backbone $BACKBONE \
    --pretrained \
    --dropout $DROPOUT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --image_size $IMAGE_SIZE \
    --samples_per_image $SAMPLES_PER_IMAGE \
    --samples_per_image_val $SAMPLES_PER_IMAGE_VAL \
    --loss_func $LOSS_FUNC \
    --scheduler $SCHEDULER \
    --identity_prob $IDENTITY_PROB \
    --cross_image_prob $CROSS_IMAGE_PROB \
    --num_workers $NUM_WORKERS \
    --seed $SEED \
    --output_dir $OUTPUT_DIR \
    --experiment_name $EXPERIMENT_NAME \
    --patience $PATIENCE \
    $WEAK_ARGS \
    "${COMET_ARGS[@]}"

echo "======================================================="
echo "g_φ trained and saved to $OUTPUT_DIR/$EXPERIMENT_NAME"
echo "======================================================="
echo ""