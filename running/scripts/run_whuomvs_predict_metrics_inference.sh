#!/bin/bash
# Run WHU-OMVS predict split metric inference for feedforward pipeline
# The predict split is the standard test set with large-frame images (3712x5504)
# Metrics follow the Ada-MVS paper (Liu et al. 2023):
#   MAE, RMSE, PAG_0.2m, PAG_0.4m, PAG_0.6m (DSM-level, Eqs. 7-9)
#   Plus standard depth metrics for cross-benchmark comparison
#
# Usage:
#   bash running/scripts/run_whuomvs_predict_metrics_inference.sh
#   bash running/scripts/run_whuomvs_predict_metrics_inference.sh --model_names depthanything3 --camera_ids 3
#   bash running/scripts/run_whuomvs_predict_metrics_inference.sh --outlier_threshold 20.0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET_PATH="${PROJECT_ROOT}/dataset/WHU-OMVS/predict"
OUTPUT_BASE="${PROJECT_ROOT}/exp/whu-omvs/predict_metric_eval"

MODEL_NAMES=("depthanything3" "mapanything" "pi3" "vggt")
CAMERA_IDS=("1" "2" "3" "4" "5")
BATCH_SIZE=1
ALIGN_MODE="median"
OUTLIER_THRESHOLD=20.0
EVAL_NORMAL=false
EVAL_DSM=true
GPU_ID=2

EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_names)       shift; IFS=',' read -ra MODEL_NAMES <<< "$1" ;;
        --camera_ids)        shift; IFS=',' read -ra CAMERA_IDS <<< "$1" ;;
        --batch_size)        shift; BATCH_SIZE="$1" ;;
        --align_mode)        shift; ALIGN_MODE="$1" ;;
        --outlier_threshold) shift; OUTLIER_THRESHOLD="$1" ;;
        --eval_normal)       shift; EVAL_NORMAL="$1" ;;
        --eval_dsm)          shift; EVAL_DSM="$1" ;;
        --gpu_id)            shift; GPU_ID="$1" ;;
        --output_base)       shift; OUTPUT_BASE="$1" ;;
        *)                   EXTRA_ARGS="${EXTRA_ARGS} $1" ;;
    esac
    shift
done

export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "============================================"
echo "WHU-OMVS Predict Split Benchmark"
echo "(Ada-MVS / Liu et al. 2023 Metrics)"
echo "============================================"
echo "Models:           ${MODEL_NAMES[*]}"
echo "Cameras:          ${CAMERA_IDS[*]}"
echo "Batch size:       ${BATCH_SIZE}"
echo "Align mode:       ${ALIGN_MODE}"
echo "Outlier T:        ${OUTLIER_THRESHOLD}m"
echo "Eval normal:      ${EVAL_NORMAL}"
echo "Eval DSM:         ${EVAL_DSM}"
echo "GPU:              ${GPU_ID}"
echo "Output:           ${OUTPUT_BASE}"
echo "============================================"

EVAL_NORMAL_FLAG=""
if [[ "${EVAL_NORMAL}" == "true" ]]; then
    EVAL_NORMAL_FLAG="--eval_normal"
fi

EVAL_DSM_FLAG=""
if [[ "${EVAL_DSM}" == "true" ]]; then
    EVAL_DSM_FLAG="--eval_dsm"
else
    EVAL_DSM_FLAG="--no_eval_dsm"
fi

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo ""
    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running predict metrics for ${MODEL_NAME}..."

    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_NAME}_predict"
    mkdir -p "${OUTPUT_DIR}"

    python3 -m running.inference.run_whuomvs_predict_metric_inference \
        --config_path "${PROJECT_ROOT}/configs/base_config.yaml" \
        --dataset_path "${DATASET_PATH}" \
        --camera_ids ${CAMERA_IDS[*]} \
        --batch_size ${BATCH_SIZE} \
        --align_mode ${ALIGN_MODE} \
        --outlier_threshold ${OUTLIER_THRESHOLD} \
        ${EVAL_NORMAL_FLAG} \
        ${EVAL_DSM_FLAG} \
        --model_name "${MODEL_NAME}" \
        --output_path "${OUTPUT_DIR}" \
        ${EXTRA_ARGS}

    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Finished ${MODEL_NAME}. Results in ${OUTPUT_DIR}"
done

echo ""
echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] All models completed."
