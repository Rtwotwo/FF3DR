#!/bin/bash
# Ada-MVS inference on WHU-OMVS predict/test split
#
# Args (edit below):
#   SPLIT           : predict / test
#   MODEL           : model name
#   LOADCKPT        : checkpoint path
#   AREAS           : area list for test split (e.g. "area2 area3")
#   EVAL_METRICS    : true/false (是否计算指标)
#   GPU_ID          : CUDA visible device id

SPLIT="test"
MODEL="adamvs"
LOADCKPT="/data2/dataset/Redal/work_feedforward_3drepo/weights/adamvs/adamvs_whuomvs/model_000019_0.1339.ckpt"
AREAS="area2 area3"
EVAL_METRICS=true
GPU_ID=2
TEST_MAX_SAMPLES_PER_CAMERA=20

VIEW_NUM=5
NUMDEPTH=192
MAX_W=3712
MAX_H=5504
RESIZE_SCALE=0.5
BATCH_SIZE=1
NDEPTHS="48,32,8"
DEPTH_INTER_R="4,2,1"
CR_BASE_CHS="8,8,8"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ "${SPLIT}" == "test" ]]; then
    DATASET_PATH="${PROJECT_ROOT}/dataset/WHU-OMVS/test"
    OUTPUT_FOLDER="${PROJECT_ROOT}/exp/whu-omvs/metric_adamvs_whuomvs_test/"
else
    DATASET_PATH="${PROJECT_ROOT}/dataset/WHU-OMVS/predict"
    OUTPUT_FOLDER="${PROJECT_ROOT}/exp/whu-omvs/metric_adamvs_whuomvs_predict/"
fi

echo "============================================"
echo "Ada-MVS Inference on WHU-OMVS ${SPLIT^^} split"
echo "============================================"
echo "  Split:       ${SPLIT}"
echo "  Areas:       ${AREAS:-auto}"
echo "  Model:       ${MODEL}"
echo "  Checkpoint:  ${LOADCKPT}"
echo "  Dataset:     ${DATASET_PATH}"
echo "  Output:      ${OUTPUT_FOLDER}"
echo "  Eval metrics:${EVAL_METRICS}"
echo "  Max samples/cam on test: ${TEST_MAX_SAMPLES_PER_CAMERA}"
echo "  GPU:         ${GPU_ID}"
echo "============================================"

ARGS=(
    --model "${MODEL}"
    --dataset "${SPLIT}_oblique"
    --output_folder "${OUTPUT_FOLDER}"
    --loadckpt "${LOADCKPT}"
    --dataset_path "${DATASET_PATH}"
    --split "${SPLIT}"
    --view_num "${VIEW_NUM}"
    --numdepth "${NUMDEPTH}"
    --max_w "${MAX_W}"
    --max_h "${MAX_H}"
    --resize_scale "${RESIZE_SCALE}"
    --batch_size "${BATCH_SIZE}"
    --ndepths "${NDEPTHS}"
    --depth_inter_r "${DEPTH_INTER_R}"
    --cr_base_chs "${CR_BASE_CHS}"
    --test_max_samples_per_camera "${TEST_MAX_SAMPLES_PER_CAMERA}"
)

if [[ "${SPLIT}" == "test" && -n "${AREAS}" ]]; then
    ARGS+=(--areas ${AREAS})
fi

if [[ "${EVAL_METRICS}" == "true" ]]; then
    ARGS+=(--eval_metrics)
fi

CUDA_VISIBLE_DEVICES=$GPU_ID python3 "${PROJECT_ROOT}/running/inference/run_whuomvs_adamvs_predict.py" "${ARGS[@]}"

echo ""
echo "[INFO] Done!"
echo "[INFO] Output folder: ${OUTPUT_FOLDER}"

if [[ "${SPLIT}" == "test" && "${EVAL_METRICS}" == "true" ]]; then
    echo ""
    echo "[INFO] Metrics files location:"
    echo "  Per-area:   ${OUTPUT_FOLDER}/{area_name}/adamvs_metrics.json"
    echo "  Combined:   ${OUTPUT_FOLDER}/test_*_all_adamvs_metrics.json"
fi
