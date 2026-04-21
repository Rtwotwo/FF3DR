#!/bin/bash
# UrbanScene NVS inference launcher
# Supports optional DA3 GS sidecar export.

DATASET_PATH="/data2/dataset/Redal/work_feedforward_3drepo/dataset/UrbanScene"
OUTPUT_ROOT="./exp/urbanscene_nvs"
RUN_ARGS_YAML="./configs/run_urbanscene_nvs_inference.yaml"
MODEL_NAME="depthanything3"
GPU_ID=6

# OOM-safe defaults for DA3 on 24GB GPU.
CHUNK_SIZE=60
OVERLAP=24
MAX_CHUNKS=-1

# DA3 GS sidecar export switches
ENABLE_DA3_GS=1
DA3_EXPORT_FORMAT="gs_ply"   # gs_ply / gs_video
DA3_EXPORT_DIR="${OUTPUT_ROOT}/da3_gs_exports"
MERGE_DA3_GS_POINTCLOUD=1
DA3_CHUNK_MAX_POINTS=2000000
DA3_MERGE_MAX_POINTS=2000000
DA3_MERGE_OUTPUT_NAME="da3_gs_merged_points.ply"

# Optional: run all UrbanScene subsets
SCENES=('PolyTech' 'ArtSci' 'School' 'Bridge' 'Castle' 'Town')
# SCENES=('PolyTech')

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

    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running UrbanScene NVS for ${SCENE} (${MODEL_NAME})..."
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$(dirname "$0")/../inference/run_urbanscene_nvs_inference.py" \
        --run_args_yaml "$RUN_ARGS_YAML" \
        --area_path "$SCENE_PATH" \
        --output_path "${OUTPUT_ROOT}/run_urbanscene_nvs_${MODEL_NAME}_${SCENE}" \
        --model_name "$MODEL_NAME" \
        --device cuda \
        --chunk_size "$CHUNK_SIZE" \
        --overlap "$OVERLAP" \
        --max_chunks "$MAX_CHUNKS" \
        --enable_da3_gs "$ENABLE_DA3_GS" \
        --da3_export_format "$DA3_EXPORT_FORMAT" \
        --da3_export_dir "${DA3_EXPORT_DIR}/${SCENE}" \
        --merge_da3_gs_pointcloud "$MERGE_DA3_GS_POINTCLOUD" \
        --da3_chunk_max_points "$DA3_CHUNK_MAX_POINTS" \
        --da3_merge_max_points "$DA3_MERGE_MAX_POINTS" \
        --da3_merge_output_name "$DA3_MERGE_OUTPUT_NAME" 
done
