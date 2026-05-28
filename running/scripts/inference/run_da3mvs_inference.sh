#!/bin/bash
# DA3MVS inference on WHU-OMVS predict/test split
#
# Args (edit below):
#   SPLIT              : predict / test
#   DA3MVS_CHECKPOINT  : fused DA3MVS checkpoint path
#   AREAS              : area list for test split (e.g. "area2 area3")
#   GPU_ID             : CUDA visible device id

SPLIT="${SPLIT:-test}"
DA3MVS_CHECKPOINT="/data2/dataset/Redal/work_feedforward_3drepo/exp/whu-omvs/train_da3mvs/da3_large_adamvs_fusion_0526/checkpoints/best.pt"
AREAS="${AREAS:-area2 area3}"
GPU_ID="${GPU_ID:-5}"

VIEW_NUM=5
BATCH_SIZE=8
OUTLIER_THRESHOLD=20.0
DEPTH_ALIGN_METHOD="median"
MULTIVIEW_NUM_NEIGHBORS=0
MULTIVIEW_PROCESS_RES=504
MULTIVIEW_REF_STRATEGY="saddle_balanced"
SAVE_ADAMVS_FORMAT=true
CAMERA_IDS=(1 2 3 4 5)
PROCESS_RES=518
PROCESS_RES_METHOD="square"

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
	DATASET_PATH="${PROJECT_ROOT}/dataset/WHU-OMVS/test"
	OUTPUT_FOLDER="${PROJECT_ROOT}/exp/whu-omvs/metric_da3mvs_whuomvs_test/"
else
	DATASET_PATH="${PROJECT_ROOT}/dataset/WHU-OMVS/predict"
	OUTPUT_FOLDER="${PROJECT_ROOT}/exp/whu-omvs/metric_da3mvs_whuomvs_predict/"
fi

echo "============================================"
echo "DA3MVS Inference on WHU-OMVS ${SPLIT^^} split"
echo "============================================"
echo "  Split:            ${SPLIT}"
echo "  Areas:            ${AREAS:-auto}"
echo "  Model:            da3mvs"
echo "  DA3MVS checkpoint:${DA3MVS_CHECKPOINT}"
echo "  Dataset:          ${DATASET_PATH}"
echo "  Output:           ${OUTPUT_FOLDER}"
echo "  Align method:     ${DEPTH_ALIGN_METHOD}"
echo "  Cameras:          ${CAMERA_IDS[*]}"
echo "  Resize:           ${PROCESS_RES} (${PROCESS_RES_METHOD})"
echo "  GPU:              ${GPU_ID}"
echo "============================================"

ARGS=(
	--config_path "${PROJECT_ROOT}/configs/base_config.yaml"
	--model_name "da3mvs"
	--da3mvs_checkpoint "${DA3MVS_CHECKPOINT}"
	--dataset_path "${DATASET_PATH}"
	--split "${SPLIT}"
	--output_path "${OUTPUT_FOLDER}"
	--camera_ids "${CAMERA_IDS[@]}"
	--batch_size "${BATCH_SIZE}"
	--align_mode median
	--outlier_threshold "${OUTLIER_THRESHOLD}"
	--depth_align_method "${DEPTH_ALIGN_METHOD}"
	--multiview_num_neighbors "${MULTIVIEW_NUM_NEIGHBORS}"
	--multiview_process_res "${MULTIVIEW_PROCESS_RES}"
	--multiview_ref_strategy "${MULTIVIEW_REF_STRATEGY}"
	--save_adamvs_format
	--no_eval_dsm
)

run_one_split() {
	local run_output_folder="$1"
	local viz_root="$2"
	local area_arg="$3"
	local save_adamvs_format_flag="${4:-true}"
	local adamvs_output_path="${run_output_folder}/adamvs_output"
	local metric_args=("${ARGS[@]}" --output_path "${run_output_folder}")
	if [[ "${save_adamvs_format_flag}" == "true" ]]; then
		metric_args+=(--adamvs_output_path "${adamvs_output_path}" --save_adamvs_format)
	fi
	if [[ -n "${area_arg}" ]]; then
		metric_args+=(--areas "${area_arg}")
	fi

	CUDA_VISIBLE_DEVICES=$GPU_ID python3 "${PROJECT_ROOT}/running/inference/run_whuomvs_dsm_metric_inference.py" "${metric_args[@]}"

	if [[ ! -d "${adamvs_output_path}" ]]; then
		echo "[WARN] Adamvs output not found: ${adamvs_output_path}"
		return 0
	fi

	python3 - <<PY
from pathlib import Path
import sys
import cv2

repo_root = Path("${PROJECT_ROOT}")
sys.path.insert(0, str(repo_root))
from running.training.datasets_adamvs.data_io import read_pfm
from running.utils.viz_utils import depth_to_color

adamvs_output = Path("${adamvs_output_path}")
viz_root = Path("${viz_root}")
split_name = "${SPLIT}"
area_name = "${area_arg}"
model_name = "da3mvs"
camera_ids = ["${CAMERA_IDS[0]}", "${CAMERA_IDS[1]}", "${CAMERA_IDS[2]}", "${CAMERA_IDS[3]}", "${CAMERA_IDS[4]}"]

if not adamvs_output.exists():
    print(f"[WARN] missing adamvs output dir: {adamvs_output}")
    raise SystemExit(0)

for cam_dir in sorted([p for p in adamvs_output.iterdir() if p.is_dir()], key=lambda p: p.name):
    cam_id = cam_dir.name
    if cam_id not in camera_ids:
        continue
    if split_name == "predict":
        out_dir = viz_root / f"run_da3mvs_{model_name}_predict_came{cam_id}"
    else:
        out_dir = viz_root / f"run_da3mvs_{model_name}_test_came{cam_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pfm_files = sorted(cam_dir.glob("*_init.pfm"))
    for pfm_path in pfm_files:
        try:
            depth, _ = read_pfm(str(pfm_path))
        except Exception as exc:
            print(f"[WARN] failed reading {pfm_path}: {exc}")
            continue
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth_color = depth_to_color(depth)
        out_png = out_dir / f"{pfm_path.stem.replace('_init', '')}_depth.png"
        cv2.imwrite(str(out_png), depth_color)

print(f"[INFO] depth visualizations saved to: {viz_root}")
PY
}

run_combined_metrics_only() {
	local run_output_folder="$1"
	local area_args=("${@:2}")
	local metric_args=("${ARGS[@]}" --output_path "${run_output_folder}")
	if [[ ${#area_args[@]} -gt 0 ]]; then
		metric_args+=(--areas "${area_args[@]}")
	fi

	CUDA_VISIBLE_DEVICES=$GPU_ID python3 "${PROJECT_ROOT}/running/inference/run_whuomvs_dsm_metric_inference.py" "${metric_args[@]}"
}

if [[ "${SPLIT}" == "predict" ]]; then
	run_one_split "${OUTPUT_FOLDER}" "${OUTPUT_FOLDER}/viz_predict" ""
else
	run_combined_metrics_only "${OUTPUT_FOLDER}" area2 area3
	for AREA in ${AREAS}; do
		run_one_split "${OUTPUT_FOLDER}/${AREA}" "${OUTPUT_FOLDER}/viz_test/${AREA}" "${AREA}" true
	done
fi

echo ""
echo "[INFO] Done!"
echo "[INFO] Output folder: ${OUTPUT_FOLDER}"