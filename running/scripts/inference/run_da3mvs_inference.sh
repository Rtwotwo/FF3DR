#!/bin/bash
# DA3MVS inference on WHU-OMVS predict/test split
#
# Args (edit below):
#   SPLIT           : predict / test
#   CHECKPOINT      : DA3MVS fused model checkpoint path
#   AREAS           : test split area list
#   GPU_ID          : CUDA visible device id

SPLIT="${SPLIT:-test}"
CHECKPOINT="${CHECKPOINT:-/data2/dataset/Redal/work_feedforward_3drepo/exp/whu-omvs/train_da3mvs/da3_large_adamvs_fusion_0526/checkpoints/best.pt}"
AREAS="${AREAS:-area2 area3}"
CAMERA_IDS="${CAMERA_IDS:-1 2 3 4 5}"
GPU_ID="${GPU_ID:-5}"
PROCESS_RES="${PROCESS_RES:-518}"
PROCESS_RES_METHOD="${PROCESS_RES_METHOD:-square}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ALIGN_MODE="${ALIGN_MODE:-median}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

SPLIT="$(echo "${SPLIT}" | tr '[:upper:]' '[:lower:]')"
if [[ "${SPLIT}" == "tests" ]]; then
	SPLIT="test"
fi
if [[ "${SPLIT}" != "predict" && "${SPLIT}" != "test" ]]; then
	echo "[ERROR] SPLIT must be 'predict' or 'test', got: ${SPLIT}"
	exit 1
fi

if [[ "${SPLIT}" == "test" ]]; then
	OUTPUT_FOLDER="${PROJECT_ROOT}/exp/whu-omvs/da3mvs_whuomvs_test"
	VIZ_MAX_FRAMES=50
else
	OUTPUT_FOLDER="${PROJECT_ROOT}/exp/whu-omvs/da3mvs_whuomvs_predict"
	VIZ_MAX_FRAMES=-1
fi

echo "============================================"
echo "DA3MVS Inference on WHU-OMVS ${SPLIT^^} split"
echo "============================================"
echo "  Split:            ${SPLIT}"
echo "  Areas:            ${AREAS:-auto}"
echo "  Camera IDs:       ${CAMERA_IDS:-auto}"
echo "  Checkpoint:       ${CHECKPOINT}"
echo "  Output:           ${OUTPUT_FOLDER}"
echo "  Resize:           ${PROCESS_RES} (${PROCESS_RES_METHOD})"
echo "  Align mode:       ${ALIGN_MODE}"
echo "  Viz max frames:   ${VIZ_MAX_FRAMES}"
echo "  Batch size:       ${BATCH_SIZE}"
echo "  GPU:              ${GPU_ID}"
echo "============================================"

mkdir -p "${OUTPUT_FOLDER}"

ARGS=(
	--split "${SPLIT}"
	--dataset_root "${PROJECT_ROOT}/dataset/WHU-OMVS"
	--output_path "${OUTPUT_FOLDER}"
	--checkpoint "${CHECKPOINT}"
	--process_res "${PROCESS_RES}"
	--process_res_method "${PROCESS_RES_METHOD}"
	--align_mode "${ALIGN_MODE}"
	--batch_size "${BATCH_SIZE}"
	--viz_max_frames "${VIZ_MAX_FRAMES}"
	--enable_viz
)

ARGS+=(--camera_ids ${CAMERA_IDS})

if [[ "${SPLIT}" == "test" && -n "${AREAS}" ]]; then
	ARGS+=(--areas ${AREAS})
fi

CUDA_VISIBLE_DEVICES=$GPU_ID python3 "${PROJECT_ROOT}/running/inference/run_whuomvs_da3mvs_inference.py" "${ARGS[@]}"

echo ""
echo "[INFO] Done!"
echo "[INFO] Output folder: ${OUTPUT_FOLDER}"