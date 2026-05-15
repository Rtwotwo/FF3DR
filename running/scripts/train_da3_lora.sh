#!/bin/bash
# DA3 LoRA Fine-tuning on MatrixCity
# ===================================
# Usage:
#   bash running/scripts/train_da3_lora.sh
#   bash running/scripts/train_da3_lora.sh --lora_r 16 --epochs 20

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_FILE="${PROJECT_ROOT}/configs/train_da3_lora.yaml"

DATASET_DIR="${PROJECT_ROOT}/dataset/MatrixCity"
CITY_SIZE="big_city"
SPLIT="train"

MODEL_NAME="da3-large"
PRETRAINED_PATH=""
LORA_R=8
LORA_ALPHA=16.0
LORA_DROPOUT=0.0

NUM_VIEWS=2
IMAGE_SIZE=504
STRIDE=10
MAX_DEPTH=500.0

BATCH_SIZE=1
NUM_WORKERS=4
EPOCHS=10
LR=1e-4
WEIGHT_DECAY=0.01
WARMUP_STEPS=500
MAX_GRAD_NORM=1.0

DEPTH_LOSS_WEIGHT=1.0
POSE_LOSS_WEIGHT=0.5

OUTPUT_DIR=""
SAVE_EVERY=1
LOG_EVERY=10
RESUME=""

GPU_ID=0
SEED=42

EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG_FILE="$2"; shift 2 ;;
        --dataset_dir)  DATASET_DIR="$2"; shift 2 ;;
        --city_size)    CITY_SIZE="$2"; shift 2 ;;
        --model_name)   MODEL_NAME="$2"; shift 2 ;;
        --pretrained_path) PRETRAINED_PATH="$2"; shift 2 ;;
        --lora_r)       LORA_R="$2"; shift 2 ;;
        --lora_alpha)   LORA_ALPHA="$2"; shift 2 ;;
        --lora_dropout) LORA_DROPOUT="$2"; shift 2 ;;
        --num_views)    NUM_VIEWS="$2"; shift 2 ;;
        --image_size)   IMAGE_SIZE="$2"; shift 2 ;;
        --stride)       STRIDE="$2"; shift 2 ;;
        --max_depth)    MAX_DEPTH="$2"; shift 2 ;;
        --batch_size)   BATCH_SIZE="$2"; shift 2 ;;
        --num_workers)  NUM_WORKERS="$2"; shift 2 ;;
        --epochs)       EPOCHS="$2"; shift 2 ;;
        --lr)           LR="$2"; shift 2 ;;
        --weight_decay) WEIGHT_DECAY="$2"; shift 2 ;;
        --warmup_steps) WARMUP_STEPS="$2"; shift 2 ;;
        --max_grad_norm) MAX_GRAD_NORM="$2"; shift 2 ;;
        --depth_loss_weight) DEPTH_LOSS_WEIGHT="$2"; shift 2 ;;
        --pose_loss_weight) POSE_LOSS_WEIGHT="$2"; shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --save_every)   SAVE_EVERY="$2"; shift 2 ;;
        --log_every)    LOG_EVERY="$2"; shift 2 ;;
        --resume)       RESUME="$2"; shift 2 ;;
        --gpu_id)       GPU_ID="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        *)              EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

echo "============================================"
echo " DA3 LoRA Fine-tuning on MatrixCity"
echo "============================================"
echo " Config:       ${CONFIG_FILE}"
echo " Dataset:      ${DATASET_DIR}/${CITY_SIZE}/${SPLIT}"
echo " Model:        ${MODEL_NAME}"
echo " LoRA:         r=${LORA_R}, alpha=${LORA_ALPHA}"
echo " Pretrained:   ${PRETRAINED_PATH:-None}"
echo " GPU:          ${GPU_ID}"
echo "============================================"

cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=${GPU_ID}

python -m running.training.train_da3_lora \
    --config "${CONFIG_FILE}" \
    --dataset_dir "${DATASET_DIR}" \
    --city_size "${CITY_SIZE}" \
    --split "${SPLIT}" \
    --model_name "${MODEL_NAME}" \
    ${PRETRAINED_PATH:+--pretrained_path "${PRETRAINED_PATH}"} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --num_views ${NUM_VIEWS} \
    --image_size ${IMAGE_SIZE} \
    --stride ${STRIDE} \
    --max_depth ${MAX_DEPTH} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_steps ${WARMUP_STEPS} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --depth_loss_weight ${DEPTH_LOSS_WEIGHT} \
    --pose_loss_weight ${POSE_LOSS_WEIGHT} \
    ${OUTPUT_DIR:+--output_dir "${OUTPUT_DIR}"} \
    --save_every ${SAVE_EVERY} \
    --log_every ${LOG_EVERY} \
    ${RESUME:+--resume "${RESUME}"} \
    --gpu_id ${GPU_ID} \
    --seed ${SEED} \
    ${EXTRA_ARGS}
