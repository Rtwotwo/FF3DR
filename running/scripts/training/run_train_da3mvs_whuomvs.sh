#!/bin/bash
# Train DA3MVS (DA3 + Ada-MVS feature fusion) on WHU-OMVS depth GT
#
# Key points:
#   - Ada-MVS feature encoder is frozen (no Ada-MVS training)
#   - Fusion uses Ada-MVS pre-depth features + DA3 DualDPT pre-head features
#   - Supervision uses WHU-OMVS train depth GT
#
# Usage:
#   bash running/scripts/training/run_train_da3mvs_whuomvs.sh
#
#   # Resume:
#   bash running/scripts/training/run_train_da3mvs_whuomvs.sh --resume exp/whu-omvs/train_da3mvs/da3_large_adamvs_fusion/checkpoints/best.pt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DATASET_ROOT="${PROJECT_ROOT}/dataset/WHU-OMVS"
OUTPUT_DIR="${PROJECT_ROOT}/exp/whu-omvs/train_da3mvs/da3_large_adamvs_fusion_0524"
MODEL_NAME="da3-large"
PRETRAINED_PATH=""
ADAMVS_CKPT="${PROJECT_ROOT}/weights/adamvs/adamvs_whuomvs/model_000019_0.1339.ckpt"
ADAMVS_FEATURE_STAGE="stage3"
FUSION_DIM=128
GPUS="5"
BATCH_SIZE=4
PROCESS_RES=504
EPOCHS=25
LR=7e-6
WEIGHT_DECAY=1e-4
WARMUP_STEPS=300
GRAD_ACCUM_STEPS=2
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="qkv proj"
ADAPTER_HIDDEN_DIM=64
ADAPTER_DEPTH_NORM=600.0
LOSS_PROFILE="depth_only"
RELATIVE_SI_WEIGHT=0.0
METRIC_SI_WEIGHT=0.0
PHASE1_METRIC_SI_WEIGHT=0.0
PHASE3_METRIC_SI_WEIGHT=0.0
LOGL1_WEIGHT=0.5
L1_WEIGHT=0.8
ABSREL_WEIGHT=0.0
GRADIENT_WEIGHT=0.08
RANGE_WEIGHT=0.0
CONFIDENCE_WEIGHT=0.0
SKY_WEIGHT=0.0
MAX_GRAD_NORM=0.8
EMA_ALPHA=0.2
EARLY_STOP_PATIENCE=6
EARLY_STOP_MIN_DELTA=0.008
CONFIDENCE_TAU=120.0
SKY_THRESHOLD=0.3
NUM_WORKERS=4
MAX_TRAIN_SAMPLES=-1
MAX_VAL_SAMPLES=500
MAX_TEST_SAMPLES=500
SAVE_EVERY_N_EPOCHS=1
SEED=42
LOG_LEVEL=INFO
RESUME=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
	case "$1" in
		--dataset_root)        shift; DATASET_ROOT="$1" ;;
		--output_dir)          shift; OUTPUT_DIR="$1" ;;
		--model_name)          shift; MODEL_NAME="$1" ;;
		--pretrained_path)     shift; PRETRAINED_PATH="$1" ;;
		--adamvs_ckpt)         shift; ADAMVS_CKPT="$1" ;;
		--adamvs_feature_stage) shift; ADAMVS_FEATURE_STAGE="$1" ;;
		--fusion_dim)          shift; FUSION_DIM="$1" ;;
		--gpus)                shift; GPUS="$1" ;;
		--batch_size)          shift; BATCH_SIZE="$1" ;;
		--process_res)         shift; PROCESS_RES="$1" ;;
		--epochs)              shift; EPOCHS="$1" ;;
		--lr)                  shift; LR="$1" ;;
		--weight_decay)        shift; WEIGHT_DECAY="$1" ;;
		--warmup_steps)        shift; WARMUP_STEPS="$1" ;;
		--grad_accum_steps)    shift; GRAD_ACCUM_STEPS="$1" ;;
		--lora_rank)           shift; LORA_RANK="$1" ;;
		--lora_alpha)          shift; LORA_ALPHA="$1" ;;
		--lora_dropout)        shift; LORA_DROPOUT="$1" ;;
		--lora_target_modules) shift; LORA_TARGET_MODULES="$1" ;;
		--adapter_hidden_dim)  shift; ADAPTER_HIDDEN_DIM="$1" ;;
		--adapter_depth_norm)  shift; ADAPTER_DEPTH_NORM="$1" ;;
		--loss_profile)        shift; LOSS_PROFILE="$1" ;;
		--relative_si_weight)  shift; RELATIVE_SI_WEIGHT="$1" ;;
		--metric_si_weight)    shift; METRIC_SI_WEIGHT="$1" ;;
		--logl1_weight)        shift; LOGL1_WEIGHT="$1" ;;
		--l1_weight)           shift; L1_WEIGHT="$1" ;;
		--absrel_weight)       shift; ABSREL_WEIGHT="$1" ;;
		--gradient_weight)     shift; GRADIENT_WEIGHT="$1" ;;
		--range_weight)        shift; RANGE_WEIGHT="$1" ;;
		--confidence_weight)   shift; CONFIDENCE_WEIGHT="$1" ;;
		--sky_weight)          shift; SKY_WEIGHT="$1" ;;
		--max_grad_norm)       shift; MAX_GRAD_NORM="$1" ;;
		--ema_alpha)           shift; EMA_ALPHA="$1" ;;
		--early_stop_patience) shift; EARLY_STOP_PATIENCE="$1" ;;
		--early_stop_min_delta) shift; EARLY_STOP_MIN_DELTA="$1" ;;
		--confidence_tau)      shift; CONFIDENCE_TAU="$1" ;;
		--sky_threshold)       shift; SKY_THRESHOLD="$1" ;;
		--num_workers)         shift; NUM_WORKERS="$1" ;;
		--max_train_samples)   shift; MAX_TRAIN_SAMPLES="$1" ;;
		--max_val_samples)     shift; MAX_VAL_SAMPLES="$1" ;;
		--max_test_samples)    shift; MAX_TEST_SAMPLES="$1" ;;
		--save_every_n_epochs) shift; SAVE_EVERY_N_EPOCHS="$1" ;;
		--seed)                shift; SEED="$1" ;;
		--log_level)           shift; LOG_LEVEL="$1" ;;
		--resume)              shift; RESUME="$1" ;;
		*)                     EXTRA_ARGS="${EXTRA_ARGS} $1" ;;
	esac
	shift
done

echo "============================================"
echo "DA3MVS (DA3 + Ada-MVS Feature Fusion)"
echo "============================================"
echo "Model:          ${MODEL_NAME}"
echo "Dataset:        ${DATASET_ROOT}"
echo "Output:         ${OUTPUT_DIR}"
echo "Ada-MVS ckpt:   ${ADAMVS_CKPT}"
echo "Ada feat stage: ${ADAMVS_FEATURE_STAGE}"
echo "Fusion dim:     ${FUSION_DIM}"
echo "GPU:            ${GPUS}"
echo "Batch size:     ${BATCH_SIZE} x grad_accum=${GRAD_ACCUM_STEPS} = effective $((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "Process res:    ${PROCESS_RES}"
echo "Epochs:         ${EPOCHS}"
echo "LR:             ${LR}, warmup=${WARMUP_STEPS}"
echo "LoRA:           rank=${LORA_RANK}, alpha=${LORA_ALPHA}, targets=${LORA_TARGET_MODULES}"
echo "Adapter:        hidden=${ADAPTER_HIDDEN_DIM}, depth_norm=${ADAPTER_DEPTH_NORM}"
echo "Profile:        ${LOSS_PROFILE}"
echo "Loss:           si=${RELATIVE_SI_WEIGHT} logl1=${LOGL1_WEIGHT} l1=${L1_WEIGHT} grad=${GRADIENT_WEIGHT} range=${RANGE_WEIGHT}"
echo "Stability:      grad_clip=${MAX_GRAD_NORM} ema_alpha=${EMA_ALPHA} early_stop=${EARLY_STOP_PATIENCE} mae_delta=${EARLY_STOP_MIN_DELTA}"
echo "Resume:         ${RESUME:-none}"
echo "============================================"

PRETRAINED_FLAG=""
if [[ -n "${PRETRAINED_PATH}" ]]; then
	PRETRAINED_FLAG="--pretrained_path ${PRETRAINED_PATH}"
fi

RESUME_FLAG=""
if [[ -n "${RESUME}" ]]; then
	RESUME_FLAG="--resume ${RESUME}"
fi

mkdir -p "${OUTPUT_DIR}"

python3 -m running.training.run_train_da3mvs_whuomvs \
	--dataset_root "${DATASET_ROOT}" \
	--output_dir "${OUTPUT_DIR}" \
	--model_name "${MODEL_NAME}" \
	${PRETRAINED_FLAG} \
	--adamvs_ckpt "${ADAMVS_CKPT}" \
	--adamvs_feature_stage ${ADAMVS_FEATURE_STAGE} \
	--fusion_dim ${FUSION_DIM} \
	--process_res ${PROCESS_RES} \
	--batch_size ${BATCH_SIZE} \
	--num_workers ${NUM_WORKERS} \
	--epochs ${EPOCHS} \
	--lr ${LR} \
	--weight_decay ${WEIGHT_DECAY} \
	--warmup_steps ${WARMUP_STEPS} \
	--grad_accum_steps ${GRAD_ACCUM_STEPS} \
	--lora_rank ${LORA_RANK} \
	--lora_alpha ${LORA_ALPHA} \
	--lora_dropout ${LORA_DROPOUT} \
	--lora_target_modules ${LORA_TARGET_MODULES} \
	--adapter_hidden_dim ${ADAPTER_HIDDEN_DIM} \
	--adapter_depth_norm ${ADAPTER_DEPTH_NORM} \
	--loss_profile ${LOSS_PROFILE} \
	--relative_si_weight ${RELATIVE_SI_WEIGHT} \
	--metric_si_weight ${METRIC_SI_WEIGHT} \
	--logl1_weight ${LOGL1_WEIGHT} \
	--l1_weight ${L1_WEIGHT} \
	--absrel_weight ${ABSREL_WEIGHT} \
	--gradient_weight ${GRADIENT_WEIGHT} \
	--range_weight ${RANGE_WEIGHT} \
	--confidence_weight ${CONFIDENCE_WEIGHT} \
	--sky_weight ${SKY_WEIGHT} \
	--max_grad_norm ${MAX_GRAD_NORM} \
	--ema_alpha ${EMA_ALPHA} \
	--early_stop_patience ${EARLY_STOP_PATIENCE} \
	--early_stop_min_delta ${EARLY_STOP_MIN_DELTA} \
	--confidence_tau ${CONFIDENCE_TAU} \
	--sky_threshold ${SKY_THRESHOLD} \
	--max_train_samples ${MAX_TRAIN_SAMPLES} \
	--max_val_samples ${MAX_VAL_SAMPLES} \
	--max_test_samples ${MAX_TEST_SAMPLES} \
	--save_every_n_epochs ${SAVE_EVERY_N_EPOCHS} \
	--seed ${SEED} \
	--log_level ${LOG_LEVEL} \
	--gpus "${GPUS}" \
	${RESUME_FLAG} \
	${EXTRA_ARGS} 2>&1 | tee "${OUTPUT_DIR}_train.log"

echo ""
echo "[INFO] Training complete. Checkpoints in: ${OUTPUT_DIR}/checkpoints/"
echo "[INFO] LoRA + Fusion weights: ${OUTPUT_DIR}/checkpoints/lora_final.pt"
echo "[INFO] Best model: ${OUTPUT_DIR}/checkpoints/best.pt"
echo ""
echo "[INFO] To monitor training with TensorBoard:"
echo "  tensorboard --logdir ${OUTPUT_DIR}/logs --port 6006 --bind_all"
