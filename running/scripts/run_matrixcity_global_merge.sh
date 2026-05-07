#!/bin/bash
# Cross-block global alignment and merging for MatrixCity.
# Supports both small_city and big_city.
#
# Args (edit below):
#   DATASET_PATH       : root of MatrixCity dataset (for VPR image extraction)
#   MODEL_NAME         : depthanything3 / mapanything / pi3 / vggt
#   CITY_SIZE          : small_city / big_city
#   TRAIN_TEST_SPLIT   : train / test
#   BASE_OUTPUT_DIR    : directory containing per-block inference outputs
#   USE_ICP_FALLBACK   : 1=use ICP when SALAD fails, 0=skip
#   MAX_CHUNKS_PER_BLOCK : max chunks to load per block (-1=all)
#   GPU_ID             : CUDA visible device id

DATASET_PATH="/data2/dataset/Redal/work_feedforward_3drepo/dataset/MatrixCity"
MODEL_NAME="depthanything3"
CITY_SIZE="big_city"
TRAIN_TEST_SPLIT="train"
BASE_OUTPUT_DIR="./exp/matrixcity"
USE_ICP_FALLBACK=1
MAX_CHUNKS_PER_BLOCK=-1
GPU_ID=0

RUN_ARGS_YAML="/data2/dataset/Redal/work_feedforward_3drepo/configs/run_matrixcity_global_merge.yaml"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MERGE_SCRIPT="${SCRIPT_DIR}/../inference/run_matrixcity_global_merge.py"

OUTPUT_PATH="${BASE_OUTPUT_DIR}/global_merge_${MODEL_NAME}_${CITY_SIZE}_${TRAIN_TEST_SPLIT}"

echo "=============================================="
echo " MatrixCity Global Block Merge"
echo " Model: ${MODEL_NAME}"
echo " City: ${CITY_SIZE}"
echo " Split: ${TRAIN_TEST_SPLIT}"
echo " Base output: ${BASE_OUTPUT_DIR}"
echo " Output: ${OUTPUT_PATH}"
echo " Config: ${RUN_ARGS_YAML}"
echo " ICP fallback: ${USE_ICP_FALLBACK}"
echo " GPU: ${GPU_ID}"
echo "=============================================="

CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$MERGE_SCRIPT" \
    --base_output_dir "$BASE_OUTPUT_DIR" \
    --city_size "$CITY_SIZE" \
    --split "$TRAIN_TEST_SPLIT" \
    --model_name "$MODEL_NAME" \
    --output_path "$OUTPUT_PATH" \
    --run_args_yaml "$RUN_ARGS_YAML" \
    --use_icp_fallback "$USE_ICP_FALLBACK" \
    --max_chunks_per_block "$MAX_CHUNKS_PER_BLOCK"

echo ""
echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Global merge finished: ${OUTPUT_PATH}/reconstruction_global.ply"
