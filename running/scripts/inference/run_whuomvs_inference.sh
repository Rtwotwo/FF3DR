#!/bin/bash
# WHU-OMVS inference (feedforward pipeline)
#
# Args (edit below):
#   TRAIN_TEST_SPLIT : predict / test / train
#   MODEL_NAME       : depthanything3 / mapanything / pi3 / vggt
#   CAMERA_IDS       : camera id list
#   CHUNK_SIZE       : frames per chunk
#   OVERLAP          : overlap between chunks
#   GPU_ID           : CUDA visible device id

TRAIN_TEST_SPLIT="test"
MODEL_NAMES=("depthanything3" 'mapanything' 'pi3' 'vggt')
CAMERA_IDS=(1 2 3 4 5)
TEST_AREAS=("area2" "area3")
CHUNK_SIZE=30
OVERLAP=12
GPU_ID=5
PROCESS_RES=518
PROCESS_RES_METHOD="square"
PROJECT_ROOT="/data2/dataset/Redal/work_feedforward_3drepo"

if [[ "${TRAIN_TEST_SPLIT}" == "predict" ]]; then
    ENABLE_VIZ=1
    VIZ_MAX_FRAMES=-1
elif [[ "${TRAIN_TEST_SPLIT}" == "test" ]]; then
    ENABLE_VIZ=1
    VIZ_MAX_FRAMES=50
else
    ENABLE_VIZ=0
    VIZ_MAX_FRAMES=-1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFERENCE_SCRIPT="${SCRIPT_DIR}/../../inference/run_whuomvs_inference.py"

echo "=============================================="
echo " WHU-OMVS Inference"
echo " Split: ${TRAIN_TEST_SPLIT}"
echo " Model: ${MODEL_NAMES}"
echo " Cameras: ${CAMERA_IDS[*]}"
echo " Chunk: ${CHUNK_SIZE}, Overlap: ${OVERLAP}"
echo " Resize: ${PROCESS_RES} (${PROCESS_RES_METHOD})"
echo " Viz: ${ENABLE_VIZ}, viz_max_frames=${VIZ_MAX_FRAMES}"
echo " GPU: ${GPU_ID}"
echo "=============================================="

run_one_case() {
    local model_name="$1"
    local camera_id="$2"
    local area_name="$3"
    local area_path=""
    local output_path=""

    if [[ "${TRAIN_TEST_SPLIT}" == "predict" ]]; then
        area_path="${PROJECT_ROOT}/dataset/WHU-OMVS/predict/Images"
        output_path="./exp/whu-omvs/viz_predict/run_whuomvs_${model_name}_${TRAIN_TEST_SPLIT}_came${camera_id}"
    elif [[ "${TRAIN_TEST_SPLIT}" == "test" ]]; then
        area_path="${PROJECT_ROOT}/dataset/WHU-OMVS/test/${area_name}/images"
        output_path="./exp/whu-omvs/viz_predict/run_whuomvs_${model_name}_${TRAIN_TEST_SPLIT}_${area_name}_came${camera_id}"
    else
        area_path="${PROJECT_ROOT}/dataset/WHU-OMVS/train/${area_name}/images"
        output_path="./exp/whu-omvs/viz_predict/run_whuomvs_${model_name}_${TRAIN_TEST_SPLIT}_${area_name}_came${camera_id}"
    fi

    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running: ${model_name} / ${area_name:-predict} / came${camera_id}..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$INFERENCE_SCRIPT" \
        --area_path "${area_path}" \
        --output_path "${output_path}" \
        --camera_ids "$camera_id" \
        --model_name "${model_name}" \
        --chunk_size ${CHUNK_SIZE} \
        --overlap ${OVERLAP} \
        --process_res ${PROCESS_RES} \
        --process_res_method "${PROCESS_RES_METHOD}" \
        --viz_max_frames ${VIZ_MAX_FRAMES} \
        $(if [[ "${ENABLE_VIZ}" -eq 1 ]]; then echo "--enable_viz"; fi)
    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Done: ${model_name} / ${area_name:-predict} / came${camera_id}"
}

for MODEL_NAME in "${MODEL_NAMES[@]}"; do 

    if [[ "${TRAIN_TEST_SPLIT}" == "test" ]]; then
        for AREA_NAME in "${TEST_AREAS[@]}"; do
            for CAME_ID in "${CAMERA_IDS[@]}"; do
                run_one_case "${MODEL_NAME}" "${CAME_ID}" "${AREA_NAME}"
            done
        done
    else
        for CAME_ID in "${CAMERA_IDS[@]}"; do
            run_one_case "${MODEL_NAME}" "${CAME_ID}" "area1"
        done
    fi
    echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] All done."
done