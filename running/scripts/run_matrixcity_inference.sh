#!/bin/bash
# MatrixCity single-view inference launcher (one block per run).


DATASET_PATH="/data2/dataset/Redal/work_feedforward_3drepo/dataset/MatrixCity"
TRAIN_TEST_SPLIT="train"
MODEL_NAME="depthanything3"
GPU_ID=6


# run reconstruction model to inference
if [ "$TRAIN_TEST_SPLIT" == "train" ]; then
    BLOCKS=('block_1' 'block_2' 'block_3' 'block_4' 'block_5' 'block_6' 'block_7' 'block_8' 'block_9' 'block_10')
    for BLOCK in "${BLOCKS[@]}"; do
        echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running inference for ${TRAIN_TEST_SPLIT} ${BLOCK}..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$(dirname "$0")/../inference/run_matrixcity_inference.py" \
            --area_path "$DATASET_PATH/small_city/aerial/${TRAIN_TEST_SPLIT}/${BLOCK}/" \
            --output_path "./exp/matrixcity/run_matrixcity_${MODEL_NAME}_${TRAIN_TEST_SPLIT}_${BLOCK}" \
            --model_name "$MODEL_NAME" \
            --chunk_size 60 \
            --overlap 24
    done
elif [ "$TRAIN_TEST_SPLIT" == "test" ]; then
    BLOCKS=('block_1_test' 'block_2_test' 'block_3_test' 'block_4_test' 'block_5_test' 'block_6_test' 'block_7_test' 'block_8_test' 'block_9_test' 'block_10_test')
    for BLOCK in "${BLOCKS[@]}"; do
        echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running inference for ${TRAIN_TEST_SPLIT} ${BLOCK}..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$(dirname "$0")/../inference/run_matrixcity_inference.py" \
            --area_path "$DATASET_PATH/small_city/aerial/${TRAIN_TEST_SPLIT}/${BLOCK}/" \
            --output_path "./exp/matrixcity/run_matrixcity_${MODEL_NAME}_${TRAIN_TEST_SPLIT}_${BLOCK}" \
            --model_name "$MODEL_NAME" \
            --chunk_size 60 \
            --overlap 24
    done
fi