#!/bin/bash
# DA3 LoRA inference on WHU-OMVS predict/test split
#
# Args (edit below):
#   SPLIT           : predict / test
#   MODEL           : model name
#   LORA_CHECKPOINT : fine-tuned DA3 LoRA checkpoint path
#   AREAS           : area list for test split (e.g. "area2 area3")
#   EVAL_DSM        : true/false (是否计算DSM指标)
#   EVAL_NORMAL     : true/false (是否计算法向量指标)
#   EVAL_RECON      : true/false (是否计算重建指标)
#   GPU_ID          : CUDA visible device id

SPLIT="predict"
MODEL="depthanything3"
LORA_CHECKPOINT="/data2/dataset/Redal/work_feedforward_3drepo/exp/train_lora_da3/da3_large_lora_whuomvs_0521/checkpoints/epoch_025.pt"
AREAS="area2 area3"
EVAL_DSM=true
EVAL_NORMAL=false
EVAL_RECON=false
GPU_ID=5

VIEW_NUM=5
BATCH_SIZE=8
OUTLIER_THRESHOLD=20.0
DEPTH_ALIGN_METHOD="median"
MULTIVIEW_NUM_NEIGHBORS=0
MULTIVIEW_PROCESS_RES=504
MULTIVIEW_REF_STRATEGY="saddle_balanced"
SAVE_ADAMVS_FORMAT=false

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ "${SPLIT}" == "test" ]]; then
	DATASET_PATH="${PROJECT_ROOT}/dataset/WHU-OMVS/test"
	OUTPUT_FOLDER="${PROJECT_ROOT}/exp/whu-omvs/metric_da3lora_whuomvs_test/"
else
	DATASET_PATH="${PROJECT_ROOT}/dataset/WHU-OMVS/predict"
	OUTPUT_FOLDER="${PROJECT_ROOT}/exp/whu-omvs/metric_da3lora_whuomvs_predict/"
fi

echo "============================================"
echo "DA3 LoRA Inference on WHU-OMVS ${SPLIT^^} split"
echo "============================================"
echo "  Split:            ${SPLIT}"
echo "  Areas:            ${AREAS:-auto}"
echo "  Model:            ${MODEL}"
echo "  LoRA checkpoint:   ${LORA_CHECKPOINT}"
echo "  Dataset:          ${DATASET_PATH}"
echo "  Output:           ${OUTPUT_FOLDER}"
echo "  Eval DSM:         ${EVAL_DSM}"
echo "  Eval normal:      ${EVAL_NORMAL}"
echo "  Eval recon:       ${EVAL_RECON}"
echo "  Align method:     ${DEPTH_ALIGN_METHOD}"
echo "  GPU:              ${GPU_ID}"
echo "============================================"

ARGS=(
	--model_name "${MODEL}"
	--dataset_path "${DATASET_PATH}"
	--split "${SPLIT}"
	--output_path "${OUTPUT_FOLDER}"
	--camera_ids 1 2 3 4 5
	--batch_size "${BATCH_SIZE}"
	--align_mode median
	--outlier_threshold "${OUTLIER_THRESHOLD}"
	--depth_align_method "${DEPTH_ALIGN_METHOD}"
	--multiview_num_neighbors "${MULTIVIEW_NUM_NEIGHBORS}"
	--multiview_process_res "${MULTIVIEW_PROCESS_RES}"
	--multiview_ref_strategy "${MULTIVIEW_REF_STRATEGY}"
	--lora_checkpoint "${LORA_CHECKPOINT}"
)

if [[ "${SPLIT}" == "test" && -n "${AREAS}" ]]; then
	ARGS+=(--areas ${AREAS})
fi

if [[ "${EVAL_DSM}" == "true" ]]; then
	ARGS+=(--eval_dsm)
else
	ARGS+=(--no_eval_dsm)
fi

if [[ "${EVAL_NORMAL}" == "true" ]]; then
	ARGS+=(--eval_normal)
fi

if [[ "${EVAL_RECON}" == "true" ]]; then
	ARGS+=(--eval_recon)
fi

if [[ "${SAVE_ADAMVS_FORMAT}" == "true" ]]; then
	ARGS+=(--save_adamvs_format)
fi

CUDA_VISIBLE_DEVICES=$GPU_ID python3 "${PROJECT_ROOT}/running/inference/run_whuomvs_dsm_metric_inference.py" "${ARGS[@]}"

echo ""
echo "[INFO] Done!"
echo "[INFO] Output folder: ${OUTPUT_FOLDER}"

if [[ "${SPLIT}" == "test" && "${EVAL_DSM}" == "true" ]]; then
	echo ""
	echo "[INFO] Metrics files location:"
	echo "  Metrics:    ${OUTPUT_FOLDER}/${SPLIT}_cam1_cam2_cam3_cam4_cam5_${MODEL}*_metrics.json"
fi
