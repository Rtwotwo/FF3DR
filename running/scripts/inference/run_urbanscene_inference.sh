#!/bin/bash
# UrbanScene inference launcher.
# Iterates over every scene folder under dataset/UrbanScene and runs the
# reconstruction pipeline scene by scene.

DATASET_PATH="/data2/dataset/Redal/work_feedforward_3drepo/dataset/UrbanScene"
MODEL_NAME="depthanything3"
GPU_ID=5
CHUNK_SIZE=60
OVERLAP=24

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFERENCE_SCRIPT="${SCRIPT_DIR}/../../inference/run_urbanscene_inference.py"
RUN_ARGS_YAML="${SCRIPT_DIR}/../../configs/run_urbanscene_inference.yaml"

if [ ! -d "$DATASET_PATH" ]; then
    echo "[ERROR] UrbanScene dataset path not found: ${DATASET_PATH}"
    exit 1
fi

if [ "$#" -gt 0 ]; then
    SCENES=()
    for SCENE_ARG in "$@"; do
        if [[ "$SCENE_ARG" = /* ]]; then
            SCENES+=("$SCENE_ARG")
        else
            SCENES+=("$DATASET_PATH/$SCENE_ARG")
        fi
    done
else
    mapfile -t SCENES < <(find "$DATASET_PATH" -mindepth 1 -maxdepth 1 -type d | sort)
fi

if [ "${#SCENES[@]}" -eq 0 ]; then
    echo "[ERROR] No scene folders found under: ${DATASET_PATH}"
    exit 1
fi

echo "=============================================="
echo " UrbanScene Inference"
echo " Model: ${MODEL_NAME}"
echo " Scenes: ${SCENES[*]}"
echo " Chunk: ${CHUNK_SIZE}, Overlap: ${OVERLAP}"
echo " GPU: ${GPU_ID}"
echo "=============================================="

for SCENE_PATH in "${SCENES[@]}"; do
    if [ ! -d "$SCENE_PATH" ]; then
        echo "[WARN $(date +"%Y-%m-%d %H:%M:%S")] Scene path not found, skipping: ${SCENE_PATH}"
        continue
    fi

    SCENE_NAME="$(basename "$SCENE_PATH")"
    OUTPUT_PATH="./exp/urbanscene/run_urbanscene_${MODEL_NAME}_${SCENE_NAME}"

    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running inference for ${SCENE_NAME}..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$INFERENCE_SCRIPT" \
        --run_args_yaml "$RUN_ARGS_YAML" \
        --area_path "$SCENE_PATH" \
        --output_path "$OUTPUT_PATH" \
        --model_name "$MODEL_NAME" \
        --chunk_size "$CHUNK_SIZE" \
        --overlap "$OVERLAP"
    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Finished: ${SCENE_NAME}"
    echo ""
done

echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] All UrbanScene scenes done."