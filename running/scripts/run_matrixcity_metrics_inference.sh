#!/bin/bash
# Run MatrixCity metric inference (one block per run)
# Args:
#   TRAIN_TEST_SPLIT: train/test
#   MODEL_NAME: depthanything3/mapanything/pi3/vggt
#   GPU_ID: available gpu id
#   ALIGN_MODE: none/median

DATASET_PATH="/data2/dataset/Redal/work_feedforward_3drepo/dataset/MatrixCity"
CONFIG_PATH="/data2/dataset/Redal/work_feedforward_3drepo/configs/base_config.yaml"
TRAIN_TEST_SPLIT="test"
MODEL_NAME="depthanything3"
ALIGN_MODE="median"
BATCH_SIZE=8
GPU_ID=4
SCENE_NAME="small_city"
VIEW_NAME="aerial"
DEPTH_ROOT="${DATASET_PATH}/${SCENE_NAME}_depth/${VIEW_NAME}"

if [[ "$TRAIN_TEST_SPLIT" == "train" ]]; then
    BLOCKS=('block_1' 'block_2' 'block_3' 'block_4' 'block_5' 'block_6' 'block_7' 'block_8' 'block_9' 'block_10')
elif [[ "$TRAIN_TEST_SPLIT" == "test" ]]; then
    BLOCKS=('block_1_test' 'block_2_test' 'block_3_test' 'block_4_test' 'block_5_test' 'block_6_test' 'block_7_test' 'block_8_test' 'block_9_test' 'block_10_test')
else
    echo "[ERROR] Unsupported split: $TRAIN_TEST_SPLIT"
    exit 1
fi

for BLOCK in "${BLOCKS[@]}"; do
    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running metrics for ${TRAIN_TEST_SPLIT}/${BLOCK} (${MODEL_NAME})..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$(dirname "$0")/../inference/run_matrixcity_metric_inference.py" \
        --run_args_yaml "./configs/run_matrixcity_metric_inference.yaml" \
        --config_path "$CONFIG_PATH" \
        --dataset_path "$DATASET_PATH" \
        --scene_name "$SCENE_NAME" \
        --view_name "$VIEW_NAME" \
        --split "$TRAIN_TEST_SPLIT" \
        --blocks "$BLOCK" \
        --batch_size "$BATCH_SIZE" \
        --align_mode "$ALIGN_MODE" \
        --model_name "$MODEL_NAME" \
        --depth_root "$DEPTH_ROOT" \
        --output_path "./exp/matrixcity/metric_eval/${MODEL_NAME}_${TRAIN_TEST_SPLIT}_${BLOCK}"
done
