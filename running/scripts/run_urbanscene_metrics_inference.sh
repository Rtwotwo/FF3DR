#!/bin/bash
# Run UrbanScene metric inference (one scene per run)
# Args:
#   MODEL_NAME: depthanything3/mapanything/pi3/vggt
#   GPU_ID: available gpu id
#   ALIGN_MODE: none/median

DATASET_PATH="/data2/dataset/Redal/work_feedforward_3drepo/dataset/UrbanScene"
CONFIG_PATH="/data2/dataset/Redal/work_feedforward_3drepo/configs/base_config.yaml"
MODEL_NAME="depthanything3"
ALIGN_MODE="median"
BATCH_SIZE=8
GPU_ID=5
SCENES=('PolyTech' 'ArtSci' 'School' 'Bridge' 'Castle' 'Town')

for SCENE in "${SCENES[@]}"; do
    SCENE_PATH="$DATASET_PATH/${SCENE}"
    if [ ! -d "$SCENE_PATH" ]; then
        echo "[WARN $(date +"%Y-%m-%d %H:%M:%S")] Skip ${SCENE}: folder not found"
        continue
    fi

    IMAGE_COUNT=$(find "$SCENE_PATH" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
    if [ "$IMAGE_COUNT" -eq 0 ]; then
        echo "[WARN $(date +"%Y-%m-%d %H:%M:%S")] Skip ${SCENE}: no images"
        continue
    fi

    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running metrics for ${SCENE} (${MODEL_NAME})..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$(dirname "$0")/../inference/run_uabanscene_metric_inference.py" \
        --run_args_yaml "./configs/run_urbanscene_metric_inference.yaml" \
        --config_path "$CONFIG_PATH" \
        --dataset_path "$DATASET_PATH" \
        --scenes "$SCENE" \
        --batch_size "$BATCH_SIZE" \
        --align_mode "$ALIGN_MODE" \
        --model_name "$MODEL_NAME" \
        --output_path "./exp/urbanscene/metric_eval/${MODEL_NAME}_${SCENE}"
done
