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
MODEL_NAMES=("depthanything3" 'mapanything' 'pi3' 'vggt')
CAMERA_IDS=(1 2 3 4 5)
CHUNK_SIZE=50
OVERLAP=25
GPU_ID=5
PROCESS_RES=518
PROCESS_RES_METHOD="square"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFERENCE_SCRIPT="${SCRIPT_DIR}/../../inference/run_whuomvs_inference.py"

echo "=============================================="
echo " WHU-OMVS Inference"
echo " Split: ${TRAIN_TEST_SPLIT}"
echo " Model: ${MODEL_NAMES}"
echo " Cameras: ${CAMERA_IDS[*]}"
echo " Chunk: ${CHUNK_SIZE}, Overlap: ${OVERLAP}"
echo " Resize: ${PROCESS_RES} (${PROCESS_RES_METHOD})"
echo " GPU: ${GPU_ID}"
echo "=============================================="

for MODEL_NAME in "${MODEL_NAMES[@]}"; do 

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
            --process_res ${PROCESS_RES} \
            --process_res_method "${PROCESS_RES_METHOD}" \
            --enable_viz
        echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Done: came${CAME_ID}"
    done
    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] All done."
done