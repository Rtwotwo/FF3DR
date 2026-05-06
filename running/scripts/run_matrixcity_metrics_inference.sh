#!/bin/bash
# Run MatrixCity metric inference (one block per run)
# Supports both small_city and big_city.
#
# Args (edit below):
#   DATASET_PATH    : root of MatrixCity dataset
#   MODEL_NAME      : depthanything3 / mapanything / pi3 / vggt
#   CITY_SIZE       : small_city / big_city
#   TRAIN_TEST_SPLIT: train / test
#   ALIGN_MODE      : none / median
#   BATCH_SIZE      : batch size for inference
#   GPU_ID          : CUDA visible device id

DATASET_PATH="/data2/dataset/Redal/work_feedforward_3drepo/dataset/MatrixCity"
CONFIG_PATH="/data2/dataset/Redal/work_feedforward_3drepo/configs/base_config.yaml"
MODEL_NAME="depthanything3"
CITY_SIZE="big_city"
TRAIN_TEST_SPLIT="test"
ALIGN_MODE="median"
BATCH_SIZE=8
GPU_ID=4

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFERENCE_SCRIPT="${SCRIPT_DIR}/../inference/run_matrixcity_metric_inference.py"

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

DEPTH_ROOT="${DATASET_PATH}/${CITY_SIZE}_depth/aerial"

echo "=============================================="
echo " MatrixCity Metric Inference"
echo " Model: ${MODEL_NAME}"
echo " City: ${CITY_SIZE}"
echo " Split: ${TRAIN_TEST_SPLIT}"
echo " Blocks: ${BLOCKS[*]}"
echo " Depth GT: ${DEPTH_ROOT}"
echo " Align: ${ALIGN_MODE}"
echo " GPU: ${GPU_ID}"
echo "=============================================="

for BLOCK in "${BLOCKS[@]}"; do
    OUTPUT_PATH="./exp/matrixcity/metric_eval/${MODEL_NAME}_${CITY_SIZE}_${TRAIN_TEST_SPLIT}_${BLOCK}"

    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running metrics for ${CITY_SIZE}/${TRAIN_TEST_SPLIT}/${BLOCK} (${MODEL_NAME})..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$INFERENCE_SCRIPT" \
        --run_args_yaml "./configs/run_matrixcity_metric_inference.yaml" \
        --config_path "$CONFIG_PATH" \
        --dataset_path "$DATASET_PATH" \
        --scene_name "$CITY_SIZE" \
        --view_name "aerial" \
        --split "$TRAIN_TEST_SPLIT" \
        --blocks "$BLOCK" \
        --batch_size "$BATCH_SIZE" \
        --align_mode "$ALIGN_MODE" \
        --model_name "$MODEL_NAME" \
        --depth_root "$DEPTH_ROOT" \
        --output_path "$OUTPUT_PATH"
    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Finished: ${BLOCK}"
    echo ""
done

echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] All blocks done."
