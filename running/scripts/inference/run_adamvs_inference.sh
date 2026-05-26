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

SPLIT="${SPLIT:-predict}"
MODEL="adamvs"
LOADCKPT="/data2/dataset/Redal/work_feedforward_3drepo/weights/adamvs/adamvs_whuomvs/model_000019_0.1339.ckpt"
AREAS="${AREAS:-area2 area3}"
EVAL_METRICS=true
DISPLAY_VIZ=true
GPU_ID="${GPU_ID:-5}"
TEST_MAX_SAMPLES_PER_CAMERA=-1
ADAMVS_TEST_MAX_SAMPLES_PER_CAMERA=20
ALIGN_MODE="median"
OUTLIER_THRESHOLD=20.0
CLEAN_OUTPUT_ON_START=false

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

# Always enable visualization and metrics; control per-split behavior below
DISPLAY_VIZ=true
EVAL_METRICS=true
if [[ "${SPLIT}" == "test" ]]; then
    DATASET_PATH="${PROJECT_ROOT}/dataset/WHU-OMVS/test"
    OUTPUT_FOLDER="${PROJECT_ROOT}/exp/whu-omvs/metric_adamvs_whuomvs_test/"
    # keep only first 50 samples per camera for test visualization
    ADAMVS_TEST_MAX_SAMPLES_PER_CAMERA=50
else
    DATASET_PATH="${PROJECT_ROOT}/dataset/WHU-OMVS/predict"
    OUTPUT_FOLDER="${PROJECT_ROOT}/exp/whu-omvs/metric_adamvs_whuomvs_predict/"
    # predict: save all samples for visualization
    ADAMVS_TEST_MAX_SAMPLES_PER_CAMERA=-1
fi

echo "============================================"
echo "Ada-MVS Inference on WHU-OMVS ${SPLIT^^} split (Unified Entry)"
echo "============================================"
echo "  Split:       ${SPLIT}"
echo "  Areas:       ${AREAS:-auto}"
echo "  Model:       ${MODEL}"
echo "  Checkpoint:  ${LOADCKPT}"
echo "  Dataset:     ${DATASET_PATH}"
echo "  Output:      ${OUTPUT_FOLDER}"
echo "  Eval metrics:${EVAL_METRICS}"
echo "  Display viz: ${DISPLAY_VIZ}"
echo "  Max samples/cam on test: ${TEST_MAX_SAMPLES_PER_CAMERA}"
echo "  Ada save cap: ${ADAMVS_TEST_MAX_SAMPLES_PER_CAMERA}"
echo "  Align mode:  ${ALIGN_MODE}"
echo "  Outlier Th:  ${OUTLIER_THRESHOLD}"
echo "  Clean output: ${CLEAN_OUTPUT_ON_START}"
echo "  GPU:         ${GPU_ID}"
echo "============================================"

if [[ "${CLEAN_OUTPUT_ON_START}" == "true" ]]; then
    if [[ -d "${OUTPUT_FOLDER}" ]]; then
        echo "[INFO] Cleaning output folder: ${OUTPUT_FOLDER}"
        rm -rf "${OUTPUT_FOLDER}"
    fi
fi

ARGS=(
    --config_path "${PROJECT_ROOT}/configs/base_config.yaml"
    --model_name "${MODEL}"
    --adamvs_ckpt "${LOADCKPT}"
    --dataset_path "${DATASET_PATH}"
    --split "${SPLIT}"
    --output_path "${OUTPUT_FOLDER}"
    --camera_ids 1 2 3 4 5
    --batch_size "${BATCH_SIZE}"
    --align_mode "${ALIGN_MODE}"
    --outlier_threshold "${OUTLIER_THRESHOLD}"
    --adamvs_test_max_samples_per_camera "${ADAMVS_TEST_MAX_SAMPLES_PER_CAMERA}"
    --save_adamvs_format
    --no_eval_dsm
)

if [[ "${SPLIT}" == "test" && -n "${AREAS}" ]]; then
    ARGS+=(--areas ${AREAS})
fi

if [[ "${EVAL_METRICS}" != "true" ]]; then
    echo "[WARN] Unified entry currently always computes depth metrics for adamvs delegation."
fi

CUDA_VISIBLE_DEVICES=$GPU_ID python3 "${PROJECT_ROOT}/running/inference/run_whuomvs_dsm_metric_inference.py" "${ARGS[@]}"

echo ""
echo "[INFO] Done!"
echo "[INFO] Output folder: ${OUTPUT_FOLDER}"

if [[ "${SPLIT}" == "test" && "${EVAL_METRICS}" == "true" ]]; then
    echo ""
    echo "[INFO] Metrics files location:"
    echo "  Per-area:   ${OUTPUT_FOLDER}/{area_name}/adamvs_metrics.json"
    echo "  Combined:   ${OUTPUT_FOLDER}/test_*_all_adamvs_metrics.json"
fi

# Post-process Ada-MVS PFMs into colored depth PNGs for easier visualization
if [[ "${DISPLAY_VIZ}" == "true" ]]; then
    ADAMVS_OUT_DIR="${OUTPUT_FOLDER}/adamvs_output"
    VIZ_OUT_DIR="${OUTPUT_FOLDER}/viz/depth_pngs"

    # Legacy Ada-MVS path writes camera folders directly under OUTPUT_FOLDER.
    if [[ ! -d "${ADAMVS_OUT_DIR}" ]]; then
        if find "${OUTPUT_FOLDER}" -maxdepth 1 -mindepth 1 -type d -regex '.*/[0-9]+' | grep -q .; then
            ADAMVS_OUT_DIR="${OUTPUT_FOLDER}"
            echo "[INFO] Using legacy Ada-MVS output layout: ${ADAMVS_OUT_DIR}"
        fi
    fi

    echo "[INFO] Converting PFMs in ${ADAMVS_OUT_DIR} -> ${VIZ_OUT_DIR}"
    python3 - <<PY
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path("${PROJECT_ROOT}").resolve()))
from running.training.datasets_adamvs.data_io import read_pfm
from running.utils.viz_utils import depth_to_color
import cv2

in_dir = Path("${ADAMVS_OUT_DIR}")
out_dir = Path("${VIZ_OUT_DIR}")
if not in_dir.exists():
    print(f"[WARN] Ada-MVS output dir not found: {in_dir}")
    raise SystemExit(0)
for root, _, files in os.walk(in_dir):
    for fn in files:
        if fn.endswith('_init.pfm'):
            src = Path(root) / fn
            rel = src.relative_to(in_dir)
            dst_dir = out_dir / rel.parent
            dst_dir.mkdir(parents=True, exist_ok=True)
            try:
                depth, _ = read_pfm(str(src))
            except Exception as e:
                print(f"[WARN] failed read_pfm {src}: {e}")
                continue
            # depth may be color or single channel; if color, take first channel
            if depth.ndim == 3:
                depth = depth[..., 0]
            color = depth_to_color(depth)
            out_path = dst_dir / (src.stem + '.png')
            cv2.imwrite(str(out_path), color)
print('[INFO] PFMs converted to PNGs')
PY
fi
