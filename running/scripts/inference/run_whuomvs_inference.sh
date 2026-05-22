#!/bin/bash
# WHU-OMVS inference (feedforward pipeline)
#
# Args (edit below):
#   TRAIN_TEST_SPLIT : predict / test / train
#   MODEL_NAME       : depthanything3 / mapanything / pi3 / vggt
#   CAMERA_IDS       : camera id list
#   CHUNK_SIZE       : frames per chunk
#   OVERLAP          : overlap between chunks
#   GPU_ID           : CUDA visible device id

TRAIN_TEST_SPLIT="predict"
MODEL_NAME="depthanything3"
CAMERA_IDS=(1 2 3 4 5)
CHUNK_SIZE=60
OVERLAP=30
GPU_ID=5

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFERENCE_SCRIPT="${SCRIPT_DIR}/../../inference/run_whuomvs_inference.py"

echo "=============================================="
echo " WHU-OMVS Inference"
echo " Split: ${TRAIN_TEST_SPLIT}"
echo " Model: ${MODEL_NAME}"
echo " Cameras: ${CAMERA_IDS[*]}"
echo " Chunk: ${CHUNK_SIZE}, Overlap: ${OVERLAP}"
echo " GPU: ${GPU_ID}"
echo "=============================================="

for CAME_ID in "${CAMERA_IDS[@]}"; do
    OUTPUT_PATH="./exp/whu-omvs/viz_predict/run_whuomvs_${MODEL_NAME}_${TRAIN_TEST_SPLIT}_came${CAME_ID}"

    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running: came${CAME_ID}..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$INFERENCE_SCRIPT" \
        --area_path "/data2/dataset/Redal/work_feedforward_3drepo/dataset/WHU-OMVS/${TRAIN_TEST_SPLIT}/Images" \
        --output_path "${OUTPUT_PATH}" \
        --camera_ids "$CAME_ID" \
        --model_name "${MODEL_NAME}" \
        --chunk_size ${CHUNK_SIZE} \
        --overlap ${OVERLAP} \
        --enable_viz
    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Done: came${CAME_ID}"
done

echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] All done."
