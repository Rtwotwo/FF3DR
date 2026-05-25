#!/bin/bash
# UrbanScene inference launcher.
#
# Args (edit below):
#   SCENES          : scene name list (决定重建顺序)
#   DATASET_PATH    : root of UrbanScene dataset
#   MODEL_NAME      : depthanything3 / mapanything / pi3 / vggt
#   GPU_ID          : CUDA visible device id
#   CHUNK_SIZE      : frames per chunk
#   OVERLAP         : overlap between chunks

DATASET_PATH="/data2/dataset/Redal/work_feedforward_3drepo/dataset/UrbanScene"
MODEL_NAMES=('mapanything' 'pi3' 'vggt')
GPU_ID=2
CHUNK_SIZE=40
OVERLAP=20

# For scenes that still drift, use anchor-stream fusion with a middle anchor camera.
DA3_INFER_MODE="anchor_stream"
ANCHOR_CAM_INDEX=2

# DA3 Gaussian splatting rendering/export. Leave OFF unless you need gsplat outputs.
ENABLE_DA3_GSPLAT=0

SCENES=('PolyTech' 'School' 'Town' 'ArtSci' 'Bridge' 'Castle' 'ArtSci_One' 'ArtSci_Two')

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFERENCE_SCRIPT="${SCRIPT_DIR}/../../inference/run_urbanscene_inference.py"
RUN_ARGS_YAML="${SCRIPT_DIR}/../../configs/run_urbanscene_inference.yaml"

echo "=============================================="
echo " UrbanScene Inference"
echo " Model: ${MODEL_NAME}"
echo " Scenes: ${SCENES[*]}"
echo " Chunk: ${CHUNK_SIZE}, Overlap: ${OVERLAP}"
echo " DA3 mode: ${DA3_INFER_MODE}, Anchor cam: ${ANCHOR_CAM_INDEX}"
echo " GPU: ${GPU_ID}"
echo "=============================================="

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for SCENE in "${SCENES[@]}"; do
        AREA_PATH="${DATASET_PATH}/${SCENE}"
        OUTPUT_PATH="./exp/urbanscene/run_urbanscene_${MODEL_NAME}_${SCENE}"

        if [ ! -d "$AREA_PATH" ]; then
            echo "[WARN $(date +"%Y-%m-%d %H:%M:%S")] Scene not found, skipping: ${SCENE}"
            continue
        fi

        echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running: ${SCENE}..."
        EXTRA_ARGS=()
        if [ "${ENABLE_DA3_GSPLAT}" = "1" ]; then
            EXTRA_ARGS+=("--enable_da3_gsplat")
        fi
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$INFERENCE_SCRIPT" \
            --run_args_yaml "$RUN_ARGS_YAML" \
            --area_path "$AREA_PATH" \
            --output_path "$OUTPUT_PATH" \
            --model_name "$MODEL_NAME" \
            --da3_infer_mode "$DA3_INFER_MODE" \
            --anchor_cam_index "$ANCHOR_CAM_INDEX" \
            --chunk_size "$CHUNK_SIZE" \
            --overlap "$OVERLAP" \
            "${EXTRA_ARGS[@]}"
        echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Done: ${SCENE}"
        echo ""
    done
    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] All scenes done."
done