#!/bin/bash
# LingBot-MAP MatrixCity inference launcher (one block per run).
# Supports both big_city and small_city.
#
# Args (edit below):
#   DATASET_PATH : root of MatrixCity dataset
#   MODEL_PATH   : path to lingbot-map checkpoint (.pt)
#   CITY_SIZE    : small_city / big_city
#   TRAIN_TEST_SPLIT : train / test
#   GPU_ID       : CUDA visible device id
#   MODE         : streaming / windowed

DATASET_PATH="/data2/dataset/Redal/work_feedforward_3drepo/dataset/MatrixCity"
MODEL_PATH="/data2/dataset/Redal/work_feedforward_3drepo/weights/lingbot-map/lingbot-map-long.pt"
CITY_SIZE="big_city"
TRAIN_TEST_SPLIT="train"
GPU_ID=0
MODE="streaming"
WINDOW_SIZE=32
OVERLAP_SIZE=8

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFERENCE_SCRIPT="${SCRIPT_DIR}/../inference/run_lingbotmap_matrixcity_inference.py"

if [ "$CITY_SIZE" == "small_city" ]; then
    if [ "$TRAIN_TEST_SPLIT" == "train" ]; then
        BLOCKS=('block_1' 'block_2' 'block_3' 'block_4' 'block_5' 'block_6' 'block_7' 'block_8' 'block_9' 'block_10')
    elif [ "$TRAIN_TEST_SPLIT" == "test" ]; then
        BLOCKS=('block_1_test' 'block_2_test' 'block_3_test' 'block_4_test' 'block_5_test' 'block_6_test' 'block_7_test' 'block_8_test' 'block_9_test' 'block_10_test')
    else
        echo "[ERROR] Unsupported split: $TRAIN_TEST_SPLIT"
        exit 1
    fi
elif [ "$CITY_SIZE" == "big_city" ]; then
    if [ "$TRAIN_TEST_SPLIT" == "train" ]; then
        BLOCKS=('big_high_block_1' 'big_high_block_2' 'big_high_block_3' 'big_high_block_4' 'big_high_block_5' 'big_high_block_6')
    elif [ "$TRAIN_TEST_SPLIT" == "test" ]; then
        BLOCKS=('big_high_block_1_test' 'big_high_block_2_test' 'big_high_block_3_test' 'big_high_block_4_test' 'big_high_block_5_test' 'big_high_block_6_test')
    else
        echo "[ERROR] Unsupported split: $TRAIN_TEST_SPLIT"
        exit 1
    fi
else
    echo "[ERROR] Unsupported city size: $CITY_SIZE (choose small_city or big_city)"
    exit 1
fi

echo "=============================================="
echo " LingBot-MAP MatrixCity Inference"
echo " City: ${CITY_SIZE}"
echo " Split: ${TRAIN_TEST_SPLIT}"
echo " Blocks: ${BLOCKS[*]}"
echo " Mode: ${MODE}"
echo " Window: ${WINDOW_SIZE}, Overlap: ${OVERLAP_SIZE}"
echo " GPU: ${GPU_ID}"
echo " Model: ${MODEL_PATH}"
echo "=============================================="

for BLOCK in "${BLOCKS[@]}"; do
    IMAGE_FOLDER="${DATASET_PATH}/${CITY_SIZE}/aerial/${TRAIN_TEST_SPLIT}/${BLOCK}/"
    OUTPUT_PATH="./exp/matrixcity/run_matrixcity_lingbotmap_${CITY_SIZE}_${TRAIN_TEST_SPLIT}_${BLOCK}"

    if [ ! -d "$IMAGE_FOLDER" ]; then
        echo "[WARN $(date +"%Y-%m-%d %H:%M:%S")] Image folder not found, skipping: ${IMAGE_FOLDER}"
        continue
    fi

    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running inference for ${CITY_SIZE}/${TRAIN_TEST_SPLIT}/${BLOCK}..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$INFERENCE_SCRIPT" \
        --model_path "$MODEL_PATH" \
        --image_folder "$IMAGE_FOLDER" \
        --output_path "$OUTPUT_PATH" \
        --mode "$MODE" \
        --window_size "$WINDOW_SIZE" \
        --overlap_size "$OVERLAP_SIZE" \
        --no_vis \
        --offload_to_cpu
    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Finished: ${BLOCK}"
    echo ""
done

echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] All blocks done."
