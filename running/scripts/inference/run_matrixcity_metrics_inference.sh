#!/bin/bash
# MatrixCity metric inference (one block per run)
#
# Args (edit below):
#   DATASET_PATH    : root of MatrixCity dataset
#   MODEL_NAME      : depthanything3 / mapanything / pi3 / vggt
#   CITY_SIZE       : small_city / big_city
#   TRAIN_TEST_SPLIT: train / test
#   BLOCKS          : block list (决定运行顺序)
#   GPU_ID          : CUDA visible device id

DATASET_PATH="/data2/dataset/Redal/work_feedforward_3drepo/dataset/MatrixCity"
MODEL_NAME="depthanything3"
CITY_SIZE="big_city"
TRAIN_TEST_SPLIT="test"
GPU_ID=0

BLOCKS=('big_high_block_1_test' 'big_high_block_2_test' 'big_high_block_3_test' \
        'big_high_block_4_test' 'big_high_block_5_test' 'big_high_block_6_test')

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFERENCE_SCRIPT="${SCRIPT_DIR}/../inference/run_matrixcity_metric_inference.py"
DEPTH_ROOT="${DATASET_PATH}/${CITY_SIZE}_depth/aerial"

echo "=============================================="
echo " MatrixCity Metric Inference"
echo " Model: ${MODEL_NAME}"
echo " City: ${CITY_SIZE}  Split: ${TRAIN_TEST_SPLIT}"
echo " Blocks: ${BLOCKS[*]}"
echo " GPU: ${GPU_ID}"
echo "=============================================="

for BLOCK in "${BLOCKS[@]}"; do
    OUTPUT_PATH="./exp/matrixcity/metric_eval/${MODEL_NAME}_${CITY_SIZE}_${TRAIN_TEST_SPLIT}_${BLOCK}"

    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running: ${BLOCK}..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$INFERENCE_SCRIPT" \
        --run_args_yaml "./configs/run_matrixcity_metric_inference.yaml" \
        --config_path "/data2/dataset/Redal/work_feedforward_3drepo/configs/base_config.yaml" \
        --dataset_path "$DATASET_PATH" \
        --scene_name "$CITY_SIZE" \
        --view_name "aerial" \
        --split "$TRAIN_TEST_SPLIT" \
        --blocks "$BLOCK" \
        --model_name "$MODEL_NAME" \
        --depth_root "$DEPTH_ROOT" \
        --output_path "$OUTPUT_PATH"
    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Done: ${BLOCK}"
done

echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] All blocks done."
