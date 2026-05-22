#!/bin/bash
# WHU-OMVS depth metric inference
#
# Args (edit below):
#   TRAIN_TEST_SPLIT : predict / test / train
#   MODEL_NAME       : depthanything3 / mapanything / pi3 / vggt
#   CAMERA_IDS       : camera id list
#   GPU_ID           : CUDA visible device id

TRAIN_TEST_SPLIT="predict"
MODEL_NAME="depthanything3"
CAMERA_IDS=(1 2 3 4 5)
GPU_ID=6

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFERENCE_SCRIPT="${SCRIPT_DIR}/../../inference/run_whuomvs_depth_metric_inference.py"

echo "=============================================="
echo " WHU-OMVS Depth Metric Inference"
echo " Split: ${TRAIN_TEST_SPLIT}"
echo " Model: ${MODEL_NAME}"
echo " Cameras: ${CAMERA_IDS[*]}"
echo " GPU: ${GPU_ID}"
echo "=============================================="

for CAME_ID in "${CAMERA_IDS[@]}"; do
    OUTPUT_PATH="./exp/whu-omvs/metric_ff3dr_depth_${TRAIN_TEST_SPLIT}/${MODEL_NAME}_${TRAIN_TEST_SPLIT}_came${CAME_ID}"

    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running: came${CAME_ID}..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$INFERENCE_SCRIPT" \
        --config_path "/data2/dataset/Redal/work_feedforward_3drepo/configs/base_config.yaml" \
        --dataset_path "/data2/dataset/Redal/work_feedforward_3drepo/dataset/WHU-OMVS" \
        --split "${TRAIN_TEST_SPLIT}" \
        --camera_id "$CAME_ID" \
        --model_name "${MODEL_NAME}" \
        --output_path "${OUTPUT_PATH}"
    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Done: came${CAME_ID}"
done

echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] All done."
