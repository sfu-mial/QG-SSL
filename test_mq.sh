#!/bin/bash
export PYTORCH_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONWARNINGS="ignore"
export TORCH_USE_RTLD_GLOBAL=YES
export CUDA_VISIBLE_DEVICES=0
export PYTHONMULTIPROCESSINGSTARTMETHOD=spawn

DATASET="PH2"
EXPERIMENT_NAME="mq_PH2"
CHECKPOINT="./checkpoints/mq/${EXPERIMENT_NAME}/best_model.pth"
TEST_CSV="./prepare_datasets/PH2_segs_metadata/test.csv"
BATCH_SIZE=32
NUM_WORKERS=4
VAL_SAMPLES=5
SEED=<SEED>

LOG_DIR="./logs/mq_test/${DATASET}_${SEED}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/test_mq_$(date +'%Y%m%d_%H%M%S').log"

exec > >(tee -a "$LOG_FILE") 2>&1

# PH2
echo "==================================================="
echo "==================================================="
echo "================ Evaluating on PH2 ================"
echo "==================================================="
echo "==================================================="
python -u training/test_mq.py \
    --checkpoint $CHECKPOINT \
    --test_csv $TEST_CSV \
    --batch_size $BATCH_SIZE \
    --samples_per_image $VAL_SAMPLES \
    --num_workers $NUM_WORKERS \
    --seed $SEED
