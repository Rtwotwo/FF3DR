#!/bin/bash
# MatrixCity single-view inference launcher (one block per run).
# Supports both small_city and big_city.
#
# Args (edit below):
#   DATASET_PATH    : root of MatrixCity dataset
#   MODEL_NAME      : depthanything3 / mapanything / pi3 / vggt
#   CITY_SIZE       : small_city / big_city
#   TRAIN_TEST_SPLIT: train / test
#   GPU_ID          : CUDA visible device id
#   CHUNK_SIZE      : frames per chunk
#   OVERLAP         : overlap between chunks

DATASET_PATH="/data2/dataset/Redal/work_feedforward_3drepo/dataset/MatrixCity"
MODEL_NAME="depthanything3"
CITY_SIZE="big_city"
TRAIN_TEST_SPLIT="train"
GPU_ID=6
CHUNK_SIZE=60
OVERLAP=24
ENABLE_GLOBAL_MERGE=1
USE_ICP_FALLBACK=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFERENCE_SCRIPT="${SCRIPT_DIR}/../inference/run_matrixcity_inference.py"
MERGE_SCRIPT="${SCRIPT_DIR}/../inference/run_matrixcity_global_merge.py"
BASE_OUTPUT_DIR="./exp/matrixcity"

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
echo " MatrixCity Inference"
echo " Model: ${MODEL_NAME}"
echo " City: ${CITY_SIZE}"
echo " Split: ${TRAIN_TEST_SPLIT}"
echo " Blocks: ${BLOCKS[*]}"
echo " Chunk: ${CHUNK_SIZE}, Overlap: ${OVERLAP}"
echo " GPU: ${GPU_ID}"
echo "=============================================="

for BLOCK in "${BLOCKS[@]}"; do
    AREA_PATH="${DATASET_PATH}/${CITY_SIZE}/aerial/${TRAIN_TEST_SPLIT}/${BLOCK}/"
    OUTPUT_PATH="./exp/matrixcity/run_matrixcity_${MODEL_NAME}_${CITY_SIZE}_${TRAIN_TEST_SPLIT}_${BLOCK}"

    if [ ! -d "$AREA_PATH" ]; then
        echo "[WARN $(date +"%Y-%m-%d %H:%M:%S")] Area path not found, skipping: ${AREA_PATH}"
        continue
    fi

    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running inference for ${CITY_SIZE}/${TRAIN_TEST_SPLIT}/${BLOCK}..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$INFERENCE_SCRIPT" \
        --area_path "$AREA_PATH" \
        --output_path "$OUTPUT_PATH" \
        --model_name "$MODEL_NAME" \
        --chunk_size "$CHUNK_SIZE" \
        --overlap "$OVERLAP"
    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Finished: ${BLOCK}"
    echo ""
done

echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] All blocks done."

if [ "$ENABLE_GLOBAL_MERGE" -eq 1 ]; then
    echo ""
    echo "=============================================="
    echo " Cross-Block Global Alignment & Merging"
    echo "=============================================="
    GLOBAL_OUTPUT="${BASE_OUTPUT_DIR}/global_merge_${MODEL_NAME}_${CITY_SIZE}_${TRAIN_TEST_SPLIT}"
    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running global merge..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$MERGE_SCRIPT" \
        --base_output_dir "$BASE_OUTPUT_DIR" \
        --city_size "$CITY_SIZE" \
        --split "$TRAIN_TEST_SPLIT" \
        --model_name "$MODEL_NAME" \
        --output_path "$GLOBAL_OUTPUT" \
        --use_icp_fallback "$USE_ICP_FALLBACK" \
        --dataset_path "$DATASET_PATH"
    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Global merge finished: ${GLOBAL_OUTPUT}/reconstruction_global.ply"
else
    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Global merge skipped (ENABLE_GLOBAL_MERGE=0)"
fi
