#!/bin/bash
# Run WHU-OMVS predict split metric inference for feedforward pipeline
# The predict split is the standard test set with large-frame images (3712x5504)
# Metrics follow the Ada-MVS paper (Liu et al. 2023):
#   MAE, RMSE, PAG_0.2m, PAG_0.4m, PAG_0.6m (DSM-level, Eqs. 7-9)
#   Plus standard depth metrics for cross-benchmark comparison
#
# Key features:
#   1. Relative-to-absolute depth alignment via GT (median / least_squares / affine)
#   2. DA3 multi-view metric depth inference (uses camera poses for metric scale)
#   3. Ada-MVS format output (PFM depth/conf, TXT cam params, JPG image)
#      enabling downstream DSM generation and metric evaluation
#
# Usage:
#   # Single-view (default, median alignment):
#   bash running/scripts/run_whuomvs_predict_metrics_inference.sh
#
#   # DA3 multi-view metric depth (5 views, no GT alignment needed):
#   bash running/scripts/run_whuomvs_predict_metrics_inference.sh \
#       --model_names depthanything3 --multiview_num_neighbors 5 \
#       --depth_align_method multiview_metric
#
#   # DA3 multi-view with affine alignment fallback:
#   bash running/scripts/run_whuomvs_predict_metrics_inference.sh \
#       --model_names depthanything3 --multiview_num_neighbors 5 \
#       --depth_align_method affine
#
#   # Specific camera:
#   bash running/scripts/run_whuomvs_predict_metrics_inference.sh --model_names depthanything3 --camera_ids 3

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
GPU_ID=0
SAVE_ADAMVS_FORMAT=true
DEPTH_ALIGN_METHOD="median"
ADAMVS_OUTPUT_PATH=""

MULTIVIEW_NUM_NEIGHBORS=0
MULTIVIEW_PROCESS_RES=504
MULTIVIEW_REF_STRATEGY="saddle_balanced"

EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_names)              shift; IFS=',' read -ra MODEL_NAMES <<< "$1" ;;
        --camera_ids)               shift; IFS=',' read -ra CAMERA_IDS <<< "$1" ;;
        --batch_size)               shift; BATCH_SIZE="$1" ;;
        --align_mode)               shift; ALIGN_MODE="$1" ;;
        --outlier_threshold)        shift; OUTLIER_THRESHOLD="$1" ;;
        --eval_normal)              shift; EVAL_NORMAL="$1" ;;
        --eval_dsm)                 shift; EVAL_DSM="$1" ;;
        --gpu_id)                   shift; GPU_ID="$1" ;;
        --output_base)              shift; OUTPUT_BASE="$1" ;;
        --save_adamvs_format)       shift; SAVE_ADAMVS_FORMAT="$1" ;;
        --depth_align_method)       shift; DEPTH_ALIGN_METHOD="$1" ;;
        --adamvs_output_path)       shift; ADAMVS_OUTPUT_PATH="$1" ;;
        --multiview_num_neighbors)  shift; MULTIVIEW_NUM_NEIGHBORS="$1" ;;
        --multiview_process_res)    shift; MULTIVIEW_PROCESS_RES="$1" ;;
        --multiview_ref_strategy)   shift; MULTIVIEW_REF_STRATEGY="$1" ;;
        *)                          EXTRA_ARGS="${EXTRA_ARGS} $1" ;;
    esac
    shift
done

export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "============================================"
echo "WHU-OMVS Predict Split Benchmark"
echo "(Ada-MVS / Liu et al. 2023 Metrics)"
echo "============================================"
echo "Models:                    ${MODEL_NAMES[*]}"
echo "Cameras:                   ${CAMERA_IDS[*]}"
echo "Batch size:                ${BATCH_SIZE}"
echo "Align mode:                ${ALIGN_MODE}"
echo "Depth align method:        ${DEPTH_ALIGN_METHOD}"
echo "Outlier T:                 ${OUTLIER_THRESHOLD}m"
echo "Eval normal:               ${EVAL_NORMAL}"
echo "Eval DSM:                  ${EVAL_DSM}"
echo "Save Ada-MVS format:       ${SAVE_ADAMVS_FORMAT}"
echo "Multi-view neighbors:      ${MULTIVIEW_NUM_NEIGHBORS}"
echo "Multi-view process res:    ${MULTIVIEW_PROCESS_RES}"
echo "Multi-view ref strategy:   ${MULTIVIEW_REF_STRATEGY}"
echo "GPU:                       ${GPU_ID}"
echo "Output:                    ${OUTPUT_BASE}"
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

SAVE_ADAMVS_FLAG=""
if [[ "${SAVE_ADAMVS_FORMAT}" == "true" ]]; then
    SAVE_ADAMVS_FLAG="--save_adamvs_format"
fi

ADAMVS_OUTPUT_FLAG=""
if [[ -n "${ADAMVS_OUTPUT_PATH}" ]]; then
    ADAMVS_OUTPUT_FLAG="--adamvs_output_path ${ADAMVS_OUTPUT_PATH}"
fi

MULTIVIEW_FLAGS=""
if [[ "${MULTIVIEW_NUM_NEIGHBORS}" -gt 0 ]]; then
    MULTIVIEW_FLAGS="--multiview_num_neighbors ${MULTIVIEW_NUM_NEIGHBORS} --multiview_process_res ${MULTIVIEW_PROCESS_RES} --multiview_ref_strategy ${MULTIVIEW_REF_STRATEGY}"
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
        --depth_align_method ${DEPTH_ALIGN_METHOD} \
        ${SAVE_ADAMVS_FLAG} \
        ${ADAMVS_OUTPUT_FLAG} \
        ${MULTIVIEW_FLAGS} \
        --model_name "${MODEL_NAME}" \
        --output_path "${OUTPUT_DIR}" \
        ${EXTRA_ARGS}

    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Finished ${MODEL_NAME}. Results in ${OUTPUT_DIR}"

    if [[ "${SAVE_ADAMVS_FORMAT}" == "true" ]]; then
        ADAMVS_DIR="${ADAMVS_OUTPUT_PATH:-${OUTPUT_DIR}/adamvs_output}"
        echo "[INFO] Ada-MVS format output saved to: ${ADAMVS_DIR}"
        echo "[INFO] Output structure per camera:"
        echo "  {cam_id}/"
        echo "    {frame_id}_init.pfm  - Metric depth map (aligned to GT scale)"
        echo "    {frame_id}_prob.pfm  - Confidence map"
        echo "    {frame_id}.txt       - Camera params (Rcw|tcw + K + depth_params)"
        echo "    {frame_id}.jpg       - Reference image"
    fi
done

echo ""
echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] All models completed."
