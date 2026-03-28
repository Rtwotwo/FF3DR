#!/bin/bash
# Run all inference scripts for 3DRepo feedforward pipeline
# Set the path to the Python interpreter and the inference script
# Args:
#     TRAIN_TEST_SPLIT: you can choose train/test/predict
#     MODEL_NAME: here are depthanything3/mapanything/pi3/vggt you can choose
#     GPU_ID: choose a available gpu to inference


DATASET_PATH="/data2/dataset/Redal/work_feedforward_3drepo/dataset/WHU-OMVS"
TRAIN_TEST_SPLIT="train"
MODEL_NAME="depthanything3"
GPU_ID=6
 

# run depth estimation inference
if [[ "$TRAIN_TEST_SPLIT" == "train" ]]; then
    # AREA_IDS=("area1" "area4" "area5" "area6")
    # CAME_IDS=(1 2 3 4 5)
    AREA_IDS=("area6")
    CAME_IDS=(1 2 3 4 5)
    for AREA_ID in "${AREA_IDS[@]}"; do
        for CAME_ID in "${CAME_IDS[@]}"; do
            echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running inference for ${AREA_ID} came${CAME_ID}..."
            CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$(dirname "$0")/../inference/run_all_inference.py" \
                --area_path "$DATASET_PATH/$TRAIN_TEST_SPLIT/$AREA_ID/images" \
                --output_path "./exp/run_whuomvs_${MODEL_NAME}_${TRAIN_TEST_SPLIT}_${AREA_ID}_came${CAME_ID}" \
                --camera_ids "$CAME_ID" \
                --model_name "$MODEL_NAME"
        done
    done
elif [[ "$TRAIN_TEST_SPLIT" == "test" ]]; then
    AREA_IDS=("area2" "area3")
    CAME_IDS=(1 2 3 4 5)
    for AREA_ID in "${AREA_IDS[@]}"; do
        for CAME_ID in "${CAME_IDS[@]}"; do
            echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running inference for ${AREA_ID} came${CAME_ID}..."
            CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$(dirname "$0")/../inference/run_all_inference.py" \
                --area_path "$DATASET_PATH/$TRAIN_TEST_SPLIT/$AREA_ID/images" \
                --output_path "./exp/run_whuomvs_${MODEL_NAME}_${TRAIN_TEST_SPLIT}_${AREA_ID}_came${CAME_ID}" \
                --camera_ids "$CAME_ID" \
                --model_name "$MODEL_NAME"
        done
    done
elif [[ "$TRAIN_TEST_SPLIT" == "predict" ]]; then
    # for predict folder, no area id, only for camera id
    CAME_IDS=(1 2 3 4 5)
    for CAME_ID in "${CAME_IDS[@]}"; do
        echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running inference for came${CAME_ID}..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$(dirname "$0")/../inference/run_all_inference.py" \
            --area_path "$DATASET_PATH/$TRAIN_TEST_SPLIT/images" \
            --output_path "./exp/run_whuomvs_${MODEL_NAME}_${TRAIN_TEST_SPLIT}_came${CAME_ID}" \
            --camera_ids "$CAME_ID" \
            --model_name "$MODEL_NAME"
    done
fi
