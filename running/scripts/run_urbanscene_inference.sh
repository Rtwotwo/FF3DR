#!/bin/bash
# UrbanScene single-view inference launcher (one scene per run).


DATASET_PATH="/data2/dataset/Redal/work_feedforward_3drepo/dataset/UrbanScene"
OUTPUT_ROOT="./exp/urbanscene"
# SCENES=('PolyTech')
SCENES=('ArtSci')
MODEL_NAME="vggt"
GPU_ID=6
 

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# run reconstruction model to inference
for SCENE in "${SCENES[@]}"; do
    SCENE_PATH="$DATASET_PATH/${SCENE}"
    if [ ! -d "$SCENE_PATH" ]; then
        echo "[WARN $(date +"%Y-%m-%d %H:%M:%S")] Skip ${SCENE}: scene folder not found (${SCENE_PATH})"
        continue
    fi

    IMAGE_COUNT=$(find "$SCENE_PATH" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
    if [ "$IMAGE_COUNT" -eq 0 ]; then
        echo "[WARN $(date +"%Y-%m-%d %H:%M:%S")] Skip ${SCENE}: no images yet (data may still be downloading)"
        continue
    fi

    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running inference for ${SCENE}..."
    if [ $MODEL_NAME != "mapanything" ]; then
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$(dirname "$0")/../inference/run_urbanscene_inference.py" \
            --run_args_yaml "./configs/run_urbanscene_inference.yaml" \
            --area_path "$SCENE_PATH" \
            --output_path "${OUTPUT_ROOT}/run_urbanscene_${MODEL_NAME}_${SCENE}" \
            --model_name "$MODEL_NAME" \
            --chunk_size 60 \
            --overlap 24
    elif [ $MODEL_NAME == "mapanything" ]; then
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$(dirname "$0")/../inference/run_urbanscene_inference.py" \
            --run_args_yaml "./configs/run_urbanscene_inference.yaml" \
            --area_path "$SCENE_PATH" \
            --output_path "${OUTPUT_ROOT}/run_urbanscene_${MODEL_NAME}_${SCENE}" \
            --model_name "$MODEL_NAME" \
            --chunk_size 30 \
            --overlap 12
    fi
done
