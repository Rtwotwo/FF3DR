#!/bin/bash
# LingBot-MAP inference launcher for multiple datasets.
# Supports MatrixCity, WHU-OMVS, and UrbanScene.
#
# Each dataset has its own independent configuration and execution section.
# Set RUN_MATRICITY / RUN_WHU_OMVS / RUN_URBANSCENE to "true" to enable.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFERENCE_SCRIPT="${SCRIPT_DIR}/../inference/run_lingbotmap_inference.py"

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Common Settings                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

DATASET_PATH="/data2/dataset/Redal/work_feedforward_3drepo/dataset"
MODEL_PATH="/data2/dataset/Redal/work_feedforward_3drepo/weights/lingbot-map/lingbot-map-long.pt"
GPU_ID=0

# ── Model ────────────────────────────────────────────────────────────────────
IMAGE_SIZE=518
PATCH_SIZE=14
ENABLE_3D_ROPE=true
MAX_FRAME_NUM=1024
NUM_SCALE_FRAMES=8
USE_SDPA=false
COMPILE=false
CAMERA_NUM_ITERATIONS=4

# ── Inference mode ───────────────────────────────────────────────────────────
MODE="streaming"
KEYFRAME_INTERVAL=""
KV_CACHE_SLIDING_WINDOW=64

# ── Windowed mode ────────────────────────────────────────────────────────────
WINDOW_SIZE=32
OVERLAP_SIZE=8

# ── Output / saving ─────────────────────────────────────────────────────────
CONF_THRESHOLD_COEF=0.75
SAMPLE_RATIO=0.015
SAVE_GLB=false
NO_VIS=true
OFFLOAD_TO_CPU=true

# ── Visualization ────────────────────────────────────────────────────────────
PORT=8080
CONF_THRESHOLD=1.5
DOWNSAMPLE_FACTOR=10
POINT_SIZE=0.00001
MASK_SKY=false
SKY_MASK_DIR=""
SKY_MASK_VIS_DIR=""
EXPORT_PREPROCESSED=""

# ── Input ────────────────────────────────────────────────────────────────────
FIRST_K=""
STRIDE=1
FPS=10

# ── Which datasets to run ────────────────────────────────────────────────────
RUN_MATRICITY=false
RUN_WHU_OMVS=false
RUN_URBANSCENE=false

# Parse command-line override: e.g. bash run_lingbotmap_inference.sh matrixcity
if [ $# -ge 1 ]; then
    case "$1" in
        matrixcity)  RUN_MATRICITY=true ;;
        whu_omvs)    RUN_WHU_OMVS=true ;;
        urbanscene)  RUN_URBANSCENE=true ;;
        all)         RUN_MATRICITY=true; RUN_WHU_OMVS=true; RUN_URBANSCENE=true ;;
        *)           echo "[ERROR] Unknown dataset: $1 (choose matrixcity, whu_omvs, urbanscene, all)"; exit 1 ;;
    esac
fi

# ── Build common argparse flags ──────────────────────────────────────────────
build_common_args() {
    local ARGS=""

    ARGS+=" --model_path ${MODEL_PATH}"
    ARGS+=" --image_size ${IMAGE_SIZE}"
    ARGS+=" --patch_size ${PATCH_SIZE}"
    ARGS+=" --mode ${MODE}"

    if [ "$ENABLE_3D_ROPE" == "true" ]; then ARGS+=" --enable_3d_rope"; fi
    ARGS+=" --max_frame_num ${MAX_FRAME_NUM}"
    ARGS+=" --num_scale_frames ${NUM_SCALE_FRAMES}"
    ARGS+=" --kv_cache_sliding_window ${KV_CACHE_SLIDING_WINDOW}"
    ARGS+=" --camera_num_iterations ${CAMERA_NUM_ITERATIONS}"

    if [ -n "$KEYFRAME_INTERVAL" ]; then ARGS+=" --keyframe_interval ${KEYFRAME_INTERVAL}"; fi
    if [ "$USE_SDPA" == "true" ];      then ARGS+=" --use_sdpa"; fi
    if [ "$COMPILE" == "true" ];       then ARGS+=" --compile"; fi
    if [ "$OFFLOAD_TO_CPU" == "true" ]; then ARGS+=" --offload_to_cpu"; else ARGS+=" --no-offload_to_cpu"; fi

    ARGS+=" --window_size ${WINDOW_SIZE}"
    ARGS+=" --overlap_size ${OVERLAP_SIZE}"

    ARGS+=" --conf_threshold_coef ${CONF_THRESHOLD_COEF}"
    ARGS+=" --sample_ratio ${SAMPLE_RATIO}"
    if [ "$SAVE_GLB" == "true" ];   then ARGS+=" --save_glb"; fi
    if [ "$NO_VIS" == "true" ];     then ARGS+=" --no_vis"; fi

    ARGS+=" --port ${PORT}"
    ARGS+=" --conf_threshold ${CONF_THRESHOLD}"
    ARGS+=" --downsample_factor ${DOWNSAMPLE_FACTOR}"
    ARGS+=" --point_size ${POINT_SIZE}"
    if [ "$MASK_SKY" == "true" ];   then ARGS+=" --mask_sky"; fi
    if [ -n "$SKY_MASK_DIR" ];      then ARGS+=" --sky_mask_dir ${SKY_MASK_DIR}"; fi
    if [ -n "$SKY_MASK_VIS_DIR" ];  then ARGS+=" --sky_mask_visualization_dir ${SKY_MASK_VIS_DIR}"; fi
    if [ -n "$EXPORT_PREPROCESSED" ]; then ARGS+=" --export_preprocessed ${EXPORT_PREPROCESSED}"; fi

    ARGS+=" --stride ${STRIDE}"
    if [ -n "$FIRST_K" ]; then ARGS+=" --first_k ${FIRST_K}"; fi

    echo "$ARGS"
}

run_inference() {
    local IMAGE_FOLDER="$1"
    local OUTPUT_PATH="$2"
    shift 2
    local DATASET_ARGS="$@"

    if [ ! -d "$IMAGE_FOLDER" ]; then
        echo "[WARN $(date +"%Y-%m-%d %H:%M:%S")] Image folder not found, skipping: ${IMAGE_FOLDER}"
        return
    fi

    local COMMON_ARGS
    COMMON_ARGS=$(build_common_args)

    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running inference: ${IMAGE_FOLDER}"
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$INFERENCE_SCRIPT" \
        --image_folder "$IMAGE_FOLDER" \
        --output_path "$OUTPUT_PATH" \
        $COMMON_ARGS \
        $DATASET_ARGS
    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Finished: ${OUTPUT_PATH}"
    echo ""
}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1. MatrixCity                                                          ║
# ║     Synthetic aerial city, single camera, transforms.json with GT poses ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── MatrixCity-specific settings ─────────────────────────────────────────────
MC_CITY_SIZE="big_city"
MC_TRAIN_TEST_SPLIT="train"
MC_IMAGE_EXT=".jpg,.png,.JPG,.PNG,.jpeg,.JPEG"

run_matrixcity() {
    echo "=============================================="
    echo " LingBot-MAP  ·  MatrixCity"
    echo " City: ${MC_CITY_SIZE}  Split: ${MC_TRAIN_TEST_SPLIT}"
    echo " Mode: ${MODE}  GPU: ${GPU_ID}"
    echo "=============================================="

    if [ "$MC_CITY_SIZE" == "small_city" ]; then
        if [ "$MC_TRAIN_TEST_SPLIT" == "train" ]; then
            BLOCKS=('block_1' 'block_2' 'block_3' 'block_4' 'block_5' 'block_6' 'block_7' 'block_8' 'block_9' 'block_10')
        elif [ "$MC_TRAIN_TEST_SPLIT" == "test" ]; then
            BLOCKS=('block_1_test' 'block_2_test' 'block_3_test' 'block_4_test' 'block_5_test' 'block_6_test' 'block_7_test' 'block_8_test' 'block_9_test' 'block_10_test')
        else
            echo "[ERROR] Unsupported split: $MC_TRAIN_TEST_SPLIT"; exit 1
        fi
    elif [ "$MC_CITY_SIZE" == "big_city" ]; then
        if [ "$MC_TRAIN_TEST_SPLIT" == "train" ]; then
            BLOCKS=('big_high_block_1' 'big_high_block_2' 'big_high_block_3' 'big_high_block_4' 'big_high_block_5' 'big_high_block_6')
        elif [ "$MC_TRAIN_TEST_SPLIT" == "test" ]; then
            BLOCKS=('big_high_block_1_test' 'big_high_block_2_test' 'big_high_block_3_test' 'big_high_block_4_test' 'big_high_block_5_test' 'big_high_block_6_test')
        else
            echo "[ERROR] Unsupported split: $MC_TRAIN_TEST_SPLIT"; exit 1
        fi
    else
        echo "[ERROR] Unsupported city size: $MC_CITY_SIZE (choose small_city or big_city)"; exit 1
    fi

    for BLOCK in "${BLOCKS[@]}"; do
        IMAGE_FOLDER="${DATASET_PATH}/MatrixCity/${MC_CITY_SIZE}/aerial/${MC_TRAIN_TEST_SPLIT}/${BLOCK}/"
        OUTPUT_PATH="./exp/matrixcity/run_lingbotmap_${MC_CITY_SIZE}_${MC_TRAIN_TEST_SPLIT}_${BLOCK}"
        run_inference "$IMAGE_FOLDER" "$OUTPUT_PATH" \
            --dataset_type matrixcity \
            --image_ext "${MC_IMAGE_EXT}"
    done
}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2. WHU-OMVS                                                            ║
# ║     Multi-camera oblique UAV, 5 cameras per frame, GT cams + depths     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── WHU-OMVS-specific settings ───────────────────────────────────────────────
WHU_SPLIT="train"
WHU_SINGLE_CAMERA=true
WHU_CAMERA_INDEX=2
WHU_CAMERA_IDS=""
WHU_NUM_CAMERAS_TO_USE=0
WHU_IMAGE_EXT=".jpg,.png,.JPG,.PNG,.jpeg,.JPEG"

run_whu_omvs() {
    echo "=============================================="
    echo " LingBot-MAP  ·  WHU-OMVS"
    echo " Split: ${WHU_SPLIT}  Single-cam: ${WHU_SINGLE_CAMERA}  Cam-index: ${WHU_CAMERA_INDEX}"
    echo " Mode: ${MODE}  GPU: ${GPU_ID}"
    echo "=============================================="

    local DATASET_ARGS="--dataset_type whu_omvs --image_ext ${WHU_IMAGE_EXT}"

    if [ "$WHU_SINGLE_CAMERA" == "true" ]; then
        DATASET_ARGS+=" --single_camera --camera_index ${WHU_CAMERA_INDEX}"
    else
        if [ -n "$WHU_CAMERA_IDS" ]; then
            DATASET_ARGS+=" --camera_ids ${WHU_CAMERA_IDS}"
        fi
        if [ "$WHU_NUM_CAMERAS_TO_USE" -gt 0 ] 2>/dev/null; then
            DATASET_ARGS+=" --num_cameras_to_use ${WHU_NUM_CAMERAS_TO_USE}"
        fi
    fi

    local AREAS=()
    local AREA_INDEX="${DATASET_PATH}/WHU-OMVS/${WHU_SPLIT}/index.txt"
    if [ -f "$AREA_INDEX" ]; then
        while IFS= read -r AREA; do
            [ -z "$AREA" ] && continue
            AREAS+=("$AREA")
        done < "$AREA_INDEX"
    else
        echo "[WARN] index.txt not found at ${AREA_INDEX}, scanning directory..."
        for D in "${DATASET_PATH}/WHU-OMVS/${WHU_SPLIT}"/*/; do
            [ -d "$D" ] && AREAS+=("$(basename "$D")")
        done
    fi

    if [ ${#AREAS[@]} -eq 0 ]; then
        echo "[ERROR] No areas found for WHU-OMVS ${WHU_SPLIT}"; exit 1
    fi

    for AREA in "${AREAS[@]}"; do
        IMAGE_FOLDER="${DATASET_PATH}/WHU-OMVS/${WHU_SPLIT}/${AREA}/"
        OUTPUT_PATH="./exp/whuomvs/run_lingbotmap_${WHU_SPLIT}_${AREA}"
        run_inference "$IMAGE_FOLDER" "$OUTPUT_PATH" $DATASET_ARGS
    done
}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3. UrbanScene                                                          ║
# ║     Real drone imagery, single camera, no GT poses                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── UrbanScene-specific settings ─────────────────────────────────────────────
US_SCENE="PolyTech"
US_IMAGE_EXT=".JPG,.jpg,.PNG,.png,.jpeg,.JPEG"

run_urbanscene() {
    echo "=============================================="
    echo " LingBot-MAP  ·  UrbanScene"
    echo " Scene: ${US_SCENE}"
    echo " Mode: ${MODE}  GPU: ${GPU_ID}"
    echo "=============================================="

    local DATASET_ARGS="--dataset_type urbanscene --image_ext ${US_IMAGE_EXT}"

    IMAGE_FOLDER="${DATASET_PATH}/UrbanScene/${US_SCENE}/"
    OUTPUT_PATH="./exp/urbanscene/run_lingbotmap_${US_SCENE}"
    run_inference "$IMAGE_FOLDER" "$OUTPUT_PATH" $DATASET_ARGS
}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Dispatch                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

echo "=============================================="
echo " LingBot-MAP Multi-Dataset Inference"
echo " MatrixCity: ${RUN_MATRICITY}"
echo " WHU-OMVS:   ${RUN_WHU_OMVS}"
echo " UrbanScene: ${RUN_URBANSCENE}"
echo " Model: ${MODEL_PATH}"
echo "=============================================="
echo ""

if [ "$RUN_MATRICITY" == "true" ]; then
    run_matrixcity
fi

if [ "$RUN_WHU_OMVS" == "true" ]; then
    run_whu_omvs
fi

if [ "$RUN_URBANSCENE" == "true" ]; then
    run_urbanscene
fi

if [ "$RUN_MATRICITY" == "false" ] && [ "$RUN_WHU_OMVS" == "false" ] && [ "$RUN_URBANSCENE" == "false" ]; then
    echo "[ERROR] No dataset selected. Usage:"
    echo "  bash $0 matrixcity   # Run MatrixCity only"
    echo "  bash $0 whu_omvs     # Run WHU-OMVS only"
    echo "  bash $0 urbanscene   # Run UrbanScene only"
    echo "  bash $0 all          # Run all three datasets"
    echo ""
    echo "Or set RUN_MATRICITY/RUN_WHU_OMVS/RUN_URBANSCENE=true in the script."
    exit 1
fi

echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] All done."
