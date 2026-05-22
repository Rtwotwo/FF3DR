#!/bin/bash
# WHU-OMVS DSM metric inference (Ada-MVS metrics)
#
# Args (edit below):
#   SPLIT           : predict / test
#   MODEL_NAMES     : model name list (决定运行顺序)
#   CAMERA_IDS      : camera id list
#   GPU_ID          : CUDA visible device id

SPLIT="test"
MODEL_NAMES=("depthanything3")
CAMERA_IDS=(1 2 3 4 5)
GPU_ID=6

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DATASET_PATH="${PROJECT_ROOT}/dataset/WHU-OMVS/${SPLIT}"
OUTPUT_BASE="${PROJECT_ROOT}/exp/whu-omvs/metric_eval_${SPLIT}"

echo "============================================"
echo " WHU-OMVS ${SPLIT^^} Split Benchmark"
echo " (Ada-MVS / Liu et al. 2023 Metrics)"
echo "============================================"
echo " Models:   ${MODEL_NAMES[*]}"
echo " Cameras:  ${CAMERA_IDS[*]}"
echo " Dataset:  ${DATASET_PATH}"
echo " Output:   ${OUTPUT_BASE}"
echo " GPU:      ${GPU_ID}"
echo "============================================"

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_NAME}_${SPLIT}"

    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running: ${MODEL_NAME}..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m running.inference.run_whuomvs_dsm_metric_inference \
        --config_path "${PROJECT_ROOT}/configs/base_config.yaml" \
        --split "${SPLIT}" \
        --dataset_path "${DATASET_PATH}" \
        --camera_ids ${CAMERA_IDS[*]} \
        --model_name "${MODEL_NAME}" \
        --output_path "${OUTPUT_DIR}" \
        --eval_dsm \
        --save_adamvs_format
    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Done: ${MODEL_NAME}"
done

echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] All done."
