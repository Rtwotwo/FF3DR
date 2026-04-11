#!/bin/bash
# UrbanScene chunked NVS inference launcher (g3splat).

DATASET_PATH="/data2/dataset/Redal/work_feedforward_3drepo/dataset/UrbanScene"
OUTPUT_ROOT="./exp/urbanscene_nvs"
SCENES=('PolyTech')
GPU_ID=5

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for SCENE in "${SCENES[@]}"; do
    SCENE_PATH="$DATASET_PATH/${SCENE}"
    if [ ! -d "$SCENE_PATH" ]; then
        echo "[WARN $(date +"%Y-%m-%d %H:%M:%S")] Skip ${SCENE}: scene folder not found (${SCENE_PATH})"
        continue
    fi
    IMAGE_COUNT=$(find "$SCENE_PATH" -maxdepth 2 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
    if [ "$IMAGE_COUNT" -lt 2 ]; then
        echo "[WARN $(date +"%Y-%m-%d %H:%M:%S")] Skip ${SCENE}: need at least 2 images (${SCENE_PATH})"
        continue
    fi

    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running chunked NVS for ${SCENE}..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$(dirname "$0")/../inference/run_urbanscene_nvs_inference.py" \
        --run_args_yaml "./configs/run_urbanscene_nvs_inference.yaml" \
        --area_path "$SCENE_PATH" \
        --output_path "${OUTPUT_ROOT}/run_urbanscene_g3splat_${SCENE}" \
        --render_batch_size 4 \
        --pair_chunk_size 120 \
        --pair_overlap 20
done
