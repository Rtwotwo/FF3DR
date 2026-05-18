#!/bin/bash
# Run Ada-MVS inference on WHU-OMVS predict split
# Outputs depth maps, confidence maps, camera params in Ada-MVS format
# Optionally evaluates depth and DSM metrics (MAE, RMSE, PAG)
#
# Usage:
#   bash running/scripts/run_adamvs_predict.sh
#   bash running/scripts/run_adamvs_predict.sh --eval_metrics
#   bash running/scripts/run_adamvs_predict.sh --gpu_id 0 --outlier_threshold 10.0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="adamvs"
DATASET="predict_oblique"
DATA_FOLDER="${PROJECT_ROOT}/dataset/WHU-OMVS/predict/source"
OUTPUT_FOLDER="${PROJECT_ROOT}/exp/adamvs_whuomvs/MVS"
LOADCKPT="${PROJECT_ROOT}/weights/adamvs/model_000014_0.1409.ckpt"
DATASET_PATH="${PROJECT_ROOT}/dataset/WHU-OMVS/predict"

VIEW_NUM=5
NUMDEPTH=192
MAX_W=3712
MAX_H=5504
RESIZE_SCALE=0.5
SAMPLE_SCALE=1
INTERVAL_SCALE=1
BATCH_SIZE=1
NDEPTHS="48,32,8"
DEPTH_INTER_R="4,2,1"
CR_BASE_CHS="8,8,8"
MIN_INTERVAL=0.1
GPU_ID=0

EVAL_METRICS=false
OUTLIER_THRESHOLD=20.0
ALIGN_MODE="none"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)              shift; MODEL="$1" ;;
        --dataset)            shift; DATASET="$1" ;;
        --data_folder)        shift; DATA_FOLDER="$1" ;;
        --output_folder)      shift; OUTPUT_FOLDER="$1" ;;
        --loadckpt)           shift; LOADCKPT="$1" ;;
        --dataset_path)       shift; DATASET_PATH="$1" ;;
        --view_num)           shift; VIEW_NUM="$1" ;;
        --numdepth)           shift; NUMDEPTH="$1" ;;
        --max_w)              shift; MAX_W="$1" ;;
        --max_h)              shift; MAX_H="$1" ;;
        --resize_scale)       shift; RESIZE_SCALE="$1" ;;
        --sample_scale)       shift; SAMPLE_SCALE="$1" ;;
        --interval_scale)     shift; INTERVAL_SCALE="$1" ;;
        --batch_size)         shift; BATCH_SIZE="$1" ;;
        --ndepths)            shift; NDEPTHS="$1" ;;
        --depth_inter_r)      shift; DEPTH_INTER_R="$1" ;;
        --cr_base_chs)        shift; CR_BASE_CHS="$1" ;;
        --min_interval)       shift; MIN_INTERVAL="$1" ;;
        --gpu_id)             shift; GPU_ID="$1" ;;
        --eval_metrics)       EVAL_METRICS=true ;;
        --outlier_threshold)  shift; OUTLIER_THRESHOLD="$1" ;;
        --align_mode)         shift; ALIGN_MODE="$1" ;;
        *)                    echo "[WARN] Unknown argument: $1" ;;
    esac
    shift
done

export CUDA_VISIBLE_DEVICES=${GPU_ID}

EVAL_FLAG=""
if [ "${EVAL_METRICS}" = "true" ]; then
    EVAL_FLAG="--eval_metrics"
fi

echo "============================================"
echo "Ada-MVS Inference on WHU-OMVS predict split"
echo "============================================"
echo "  Model:          ${MODEL}"
echo "  Checkpoint:     ${LOADCKPT}"
echo "  Data folder:    ${DATA_FOLDER}"
echo "  Output folder:  ${OUTPUT_FOLDER}"
echo "  GPU:            ${GPU_ID}"
echo "  Eval metrics:   ${EVAL_METRICS}"
echo "============================================"

cd "${PROJECT_ROOT}"

python3 running/inference/run_whuomvs_adamvs_predict.py \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --data_folder ${DATA_FOLDER} \
    --output_folder ${OUTPUT_FOLDER} \
    --loadckpt ${LOADCKPT} \
    --dataset_path ${DATASET_PATH} \
    --view_num ${VIEW_NUM} \
    --numdepth ${NUMDEPTH} \
    --max_w ${MAX_W} \
    --max_h ${MAX_H} \
    --resize_scale ${RESIZE_SCALE} \
    --sample_scale ${SAMPLE_SCALE} \
    --interval_scale ${INTERVAL_SCALE} \
    --batch_size ${BATCH_SIZE} \
    --ndepths ${NDEPTHS} \
    --depth_inter_r ${DEPTH_INTER_R} \
    --cr_base_chs ${CR_BASE_CHS} \
    --min_interval ${MIN_INTERVAL} \
    --outlier_threshold ${OUTLIER_THRESHOLD} \
    --align_mode ${ALIGN_MODE} \
    ${EVAL_FLAG}

echo ""
echo
