#!/bin/bash
# LingBot-MAP inference launcher
#
# Args (edit below):
#   DATASET_TYPE    : matrixcity / whu_omvs / urbanscene
#   IMAGE_FOLDER    : path to images
#   MODEL_PATH      : model checkpoint path
#   GPU_ID          : CUDA visible device id

DATASET_TYPE="urbanscene"
IMAGE_FOLDER="/data2/dataset/Redal/work_feedforward_3drepo/dataset/UrbanScene/PolyTech"
MODEL_PATH="/data2/dataset/Redal/work_feedforward_3drepo/weights/lingbot-map/lingbot-map-long.pt"
GPU_ID=0

IMAGE_SIZE=518
MODE="streaming"
WINDOW_SIZE=32
OVERLAP_SIZE=8
CONF_THRESHOLD_COEF=0.75
SAMPLE_RATIO=0.015

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFERENCE_SCRIPT="${SCRIPT_DIR}/../inference/run_lingbotmap_inference.py"

OUTPUT_PATH="./exp/urbanscene/run_lingbotmap_$(basename "$IMAGE_FOLDER")"

echo "=============================================="
echo " LingBot-MAP Inference"
echo " Dataset: ${DATASET_TYPE}"
echo " Images:  ${IMAGE_FOLDER}"
echo " Model:   ${MODEL_PATH}"
echo " GPU:     ${GPU_ID}"
echo "=============================================="

if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "[ERROR] Image folder not found: ${IMAGE_FOLDER}"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$INFERENCE_SCRIPT" \
    --image_folder "$IMAGE_FOLDER" \
    --output_path "$OUTPUT_PATH" \
    --model_path "${MODEL_PATH}" \
    --dataset_type "${DATASET_TYPE}" \
    --image_size ${IMAGE_SIZE} \
    --mode "${MODE}" \
    --window_size ${WINDOW_SIZE} \
    --overlap_size ${OVERLAP_SIZE} \
    --conf_threshold_coef ${CONF_THRESHOLD_COEF} \
    --sample_ratio ${SAMPLE_RATIO} \
    --no_vis \
    --offload_to_cpu

echo "[INFO] Done. Output at: ${OUTPUT_PATH}"
