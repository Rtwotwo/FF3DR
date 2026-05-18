#!/bin/bash
# Fine-tune DA3 with LoRA for metric depth on WHU-OMVS dataset
#
# Strategy:
#   1. LoRA (rank=8) on ViT backbone qkv/proj layers (~0.8M params)
#   2. Full fine-tuning on DPT depth head (~50M params)
#   3. Metric depth loss: SI-log + L1 + affine regularization + gradient matching
#   4. Affine regularization forces scale→1 and shift→0, eliminating the
#      GT = a*pred + b affine distortion problem
#
# Memory: ~6-8GB on RTX 3090 (batch_size=2, process_res=504)
#
# Usage:
#   bash running/scripts/train_da3_lora_whuomvs.sh
#
#   # Custom GPU:
#   bash running/scripts/train_da3_lora_whuomvs.sh --gpu_id 5
#
#   # Resume from checkpoint:
#   bash running/scripts/train_da3_lora_whuomvs.sh --resume exp/da3_lora_whuomvs/checkpoints/best.pt

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET_ROOT="${PROJECT_ROOT}/dataset/WHU-OMVS"
OUTPUT_DIR="${PROJECT_ROOT}/exp/train_da3_large_lora_whuomvs"
MODEL_NAME="da3-large"
PRETRAINED_PATH=""
GPU_ID=0
BATCH_SIZE=2
PROCESS_RES=504
EPOCHS=25
LR=1e-4
WEIGHT_DECAY=1e-4
WARMUP_STEPS=500
GRAD_ACCUM_STEPS=4
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
SI_WEIGHT=1.0
L1_WEIGHT=1.0
AFFINE_WEIGHT=0.5
GRADIENT_WEIGHT=0.1
NUM_WORKERS=4
MAX_TRAIN_SAMPLES=-1
MAX_VAL_SAMPLES=500
SAVE_EVERY_N_EPOCHS=5
SEED=42
LOG_LEVEL=INFO
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset_root)        shift; DATASET_ROOT="$1" ;;
        --output_dir)          shift; OUTPUT_DIR="$1" ;;
        --model_name)          shift; MODEL_NAME="$1" ;;
        --pretrained_path)     shift; PRETRAINED_PATH="$1" ;;
        --gpu_id)              shift; GPU_ID="$1" ;;
        --batch_size)          shift; BATCH_SIZE="$1" ;;
        --process_res)         shift; PROCESS_RES="$1" ;;
        --epochs)              shift; EPOCHS="$1" ;;
        --lr)                  shift; LR="$1" ;;
        --weight_decay)        shift; WEIGHT_DECAY="$1" ;;
        --warmup_steps)        shift; WARMUP_STEPS="$1" ;;
        --grad_accum_steps)    shift; GRAD_ACCUM_STEPS="$1" ;;
        --lora_rank)           shift; LORA_RANK="$1" ;;
        --lora_alpha)          shift; LORA_ALPHA="$1" ;;
        --lora_dropout)        shift; LORA_DROPOUT="$1" ;;
        --si_weight)           shift; SI_WEIGHT="$1" ;;
        --l1_weight)           shift; L1_WEIGHT="$1" ;;
        --affine_weight)       shift; AFFINE_WEIGHT="$1" ;;
        --gradient_weight)     shift; GRADIENT_WEIGHT="$1" ;;
        --num_workers)         shift; NUM_WORKERS="$1" ;;
        --max_train_samples)   shift; MAX_TRAIN_SAMPLES="$1" ;;
        --max_val_samples)     shift; MAX_VAL_SAMPLES="$1" ;;
        --save_every_n_epochs) shift; SAVE_EVERY_N_EPOCHS="$1" ;;
        --seed)                shift; SEED="$1" ;;
        --log_level)           shift; LOG_LEVEL="$1" ;;
        *)                     EXTRA_ARGS="${EXTRA_ARGS} $1" ;;
    esac
    shift
done

export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "============================================"
echo "DA3 LoRA Fine-tuning for Metric Depth"
echo "============================================"
echo "Model:          ${MODEL_NAME}"
echo "Dataset:        ${DATASET_ROOT}"
echo "Output:         ${OUTPUT_DIR}"
echo "GPU:            ${GPU_ID}"
echo "Batch size:     ${BATCH_SIZE}"
echo "Process res:    ${PROCESS_RES}"
echo "Epochs:         ${EPOCHS}"
echo "LR:             ${LR}"
echo "LoRA rank:      ${LORA_RANK}"
echo "LoRA alpha:     ${LORA_ALPHA}"
echo "Grad accum:     ${GRAD_ACCUM_STEPS}"
echo "Loss weights:   SI=${SI_WEIGHT} L1=${L1_WEIGHT} Affine=${AFFINE_WEIGHT} Grad=${GRADIENT_WEIGHT}"
echo "============================================"

PRETRAINED_FLAG=""
if [[ -n "${PRETRAINED_PATH}" ]]; then
    PRETRAINED_FLAG="--pretrained_path ${PRETRAINED_PATH}"
fi

python3 -m running.training.train_da3_lora_whuomvs \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_name "${MODEL_NAME}" \
    ${PRETRAINED_FLAG} \
    --process_res ${PROCESS_RES} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_steps ${WARMUP_STEPS} \
    --grad_accum_steps ${GRAD_ACCUM_STEPS} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --si_weight ${SI_WEIGHT} \
    --l1_weight ${L1_WEIGHT} \
    --affine_weight ${AFFINE_WEIGHT} \
    --gradient_weight ${GRADIENT_WEIGHT} \
    --max_train_samples ${MAX_TRAIN_SAMPLES} \
    --max_val_samples ${MAX_VAL_SAMPLES} \
    --save_every_n_epochs ${SAVE_EVERY_N_EPOCHS} \
    --seed ${SEED} \
    --log_level ${LOG_LEVEL} \
    --gpu_id ${GPU_ID} \
    ${EXTRA_ARGS}

echo ""
echo "[INFO] Training complete. Checkpoints in: ${OUTPUT_DIR}/checkpoints/"
echo "[INFO] LoRA weights: ${OUTPUT_DIR}/checkpoints/lora_final.pt"
echo "[INFO] Best model: ${OUTPUT_DIR}/checkpoints/best.pt"
echo ""
echo "[INFO] To monitor training with TensorBoard:"
echo "  tensorboard --logdir ${OUTPUT_DIR}/logs --port 6006 --bind_all"
