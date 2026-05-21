#!/bin/bash
# Fine-tune DA3-Large with LoRA + MetricAdapterV3 for metric depth on WHU-OMVS
#
# Key improvements over v2:
#   - MetricAdapterV3: CNN-based per-pixel residual correction
#     pred_metric = pred * exp(log_scale) + shift + CNN(pred_norm) * depth_norm
#   - Log-depth L1 loss (dominant) + SI-log + L1 + multi-scale gradient + range constraint
#   - 3-phase training: Phase1(adapter only) -> Phase2(adapter+LoRA) -> Phase3(adapter+LoRA+head)
#   - Validation on both val split (area1) and test split (area2+area3)
#   - TensorBoard: epoch/lr/train+val+test curves
#   - Print every 500 steps
#   - Single GPU (GPU 0, 24GB free), batch_size=2, grad_accum=4
#
# Memory estimate: ~16-18GB for da3-large + LoRA + adapter on RTX 3090
#
# Usage:
#   bash running/scripts/training/run_train_da3_lora_whuomvs.sh
#
#   # Resume from checkpoint:
#   bash running/scripts/training/run_train_da3_lora_whuomvs.sh --resume exp/da3_large_lora_whuomvs/checkpoints/best.pt

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DATASET_ROOT="${PROJECT_ROOT}/dataset/WHU-OMVS"
OUTPUT_DIR="${PROJECT_ROOT}/exp/da3_large_lora_whuomvs"
MODEL_NAME="da3-large"
PRETRAINED_PATH=""
GPUS="7"
BATCH_SIZE=2
PROCESS_RES=504
EPOCHS=30
LR=5e-5
WEIGHT_DECAY=1e-4
WARMUP_STEPS=500
GRAD_ACCUM_STEPS=4
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="qkv proj"
PHASE1_EPOCHS=2
PHASE2_EPOCHS=10
ADAPTER_HIDDEN_DIM=64
ADAPTER_DEPTH_NORM=600.0
SI_WEIGHT=1.0
LOGL1_WEIGHT=10.0
L1_WEIGHT=1.0
ABSREL_WEIGHT=0.5
GRADIENT_WEIGHT=0.5
RANGE_WEIGHT=0.1
SCALE_REG_WEIGHT=0.01
SHIFT_REG_WEIGHT=0.01
NUM_WORKERS=4
MAX_TRAIN_SAMPLES=-1
MAX_VAL_SAMPLES=500
MAX_TEST_SAMPLES=500
SAVE_EVERY_N_EPOCHS=5
VAL_INTERVAL_STEPS=0
PRINT_EVERY_STEPS=500
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
        --phase1_epochs)       shift; PHASE1_EPOCHS="$1" ;;
        --phase2_epochs)       shift; PHASE2_EPOCHS="$1" ;;
        --adapter_hidden_dim)  shift; ADAPTER_HIDDEN_DIM="$1" ;;
        --adapter_depth_norm)  shift; ADAPTER_DEPTH_NORM="$1" ;;
        --si_weight)           shift; SI_WEIGHT="$1" ;;
        --logl1_weight)        shift; LOGL1_WEIGHT="$1" ;;
        --l1_weight)           shift; L1_WEIGHT="$1" ;;
        --absrel_weight)       shift; ABSREL_WEIGHT="$1" ;;
        --gradient_weight)     shift; GRADIENT_WEIGHT="$1" ;;
        --range_weight)        shift; RANGE_WEIGHT="$1" ;;
        --scale_reg_weight)    shift; SCALE_REG_WEIGHT="$1" ;;
        --shift_reg_weight)    shift; SHIFT_REG_WEIGHT="$1" ;;
        --num_workers)         shift; NUM_WORKERS="$1" ;;
        --max_train_samples)   shift; MAX_TRAIN_SAMPLES="$1" ;;
        --max_val_samples)     shift; MAX_VAL_SAMPLES="$1" ;;
        --max_test_samples)    shift; MAX_TEST_SAMPLES="$1" ;;
        --save_every_n_epochs) shift; SAVE_EVERY_N_EPOCHS="$1" ;;
        --val_interval_steps)  shift; VAL_INTERVAL_STEPS="$1" ;;
        --print_every_steps)   shift; PRINT_EVERY_STEPS="$1" ;;
        --seed)                shift; SEED="$1" ;;
        --log_level)           shift; LOG_LEVEL="$1" ;;
        --resume)              shift; RESUME="$1" ;;
        *)                     EXTRA_ARGS="${EXTRA_ARGS} $1" ;;
    esac
    shift
done

PHASE3_EPOCHS=$((EPOCHS - PHASE1_EPOCHS - PHASE2_EPOCHS))

echo "============================================"
echo "DA3-Large LoRA + MetricAdapterV3 Fine-tuning"
echo "============================================"
echo "Model:          ${MODEL_NAME}"
echo "Dataset:        ${DATASET_ROOT}"
echo "Output:         ${OUTPUT_DIR}"
echo "GPU:            ${GPUS}"
echo "Batch size:     ${BATCH_SIZE} x grad_accum=${GRAD_ACCUM_STEPS} = effective $((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "Process res:    ${PROCESS_RES}"
echo "Epochs:         ${EPOCHS} (P1=${PHASE1_EPOCHS}, P2=${PHASE2_EPOCHS}, P3=${PHASE3_EPOCHS})"
echo "LR:             ${LR}, warmup=${WARMUP_STEPS}"
echo "LoRA:           rank=${LORA_RANK}, alpha=${LORA_ALPHA}, targets=${LORA_TARGET_MODULES}"
echo "Adapter:        hidden=${ADAPTER_HIDDEN_DIM}, depth_norm=${ADAPTER_DEPTH_NORM}"
echo "Loss:           SI=${SI_WEIGHT} LogL1=${LOGL1_WEIGHT} L1=${L1_WEIGHT} Grad=${GRADIENT_WEIGHT} Range=${RANGE_WEIGHT}"
echo "Loss+Reg:       AbsRel=${ABSREL_WEIGHT} ScaleReg=${SCALE_REG_WEIGHT} ShiftReg=${SHIFT_REG_WEIGHT}"
if [[ "${VAL_INTERVAL_STEPS}" -gt 0 ]]; then
    echo "Val interval:   every ${VAL_INTERVAL_STEPS} steps"
else
    echo "Val interval:   epoch-end only (val_interval_steps=0)"
fi
echo "Print interval: every ${PRINT_EVERY_STEPS} steps"
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

python3 -m running.training.run_train_da3_lora_whuomvs \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_name "${MODEL_NAME}" \
    ${PRETRAINED_FLAG} \
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
    --phase1_epochs ${PHASE1_EPOCHS} \
    --phase2_epochs ${PHASE2_EPOCHS} \
    --adapter_hidden_dim ${ADAPTER_HIDDEN_DIM} \
    --adapter_depth_norm ${ADAPTER_DEPTH_NORM} \
    --si_weight ${SI_WEIGHT} \
    --logl1_weight ${LOGL1_WEIGHT} \
    --l1_weight ${L1_WEIGHT} \
    --absrel_weight ${ABSREL_WEIGHT} \
    --gradient_weight ${GRADIENT_WEIGHT} \
    --range_weight ${RANGE_WEIGHT} \
    --scale_reg_weight ${SCALE_REG_WEIGHT} \
    --shift_reg_weight ${SHIFT_REG_WEIGHT} \
    --max_train_samples ${MAX_TRAIN_SAMPLES} \
    --max_val_samples ${MAX_VAL_SAMPLES} \
    --max_test_samples ${MAX_TEST_SAMPLES} \
    --save_every_n_epochs ${SAVE_EVERY_N_EPOCHS} \
    --val_interval_steps ${VAL_INTERVAL_STEPS} \
    --print_every_steps ${PRINT_EVERY_STEPS} \
    --seed ${SEED} \
    --log_level ${LOG_LEVEL} \
    --gpus "${GPUS}" \
    ${RESUME_FLAG} \
    ${EXTRA_ARGS} 2>&1 | tee "${OUTPUT_DIR}_train.log"

echo ""
echo "[INFO] Training complete. Checkpoints in: ${OUTPUT_DIR}/checkpoints/"
echo "[INFO] MetricAdapter weights: ${OUTPUT_DIR}/checkpoints/metric_adapter.pt"
echo "[INFO] LoRA + Adapter weights: ${OUTPUT_DIR}/checkpoints/lora_final.pt"
echo "[INFO] Best model: ${OUTPUT_DIR}/checkpoints/best.pt"
echo ""
echo "[INFO] To monitor training with TensorBoard:"
echo "  tensorboard --logdir ${OUTPUT_DIR}/logs --port 6006 --bind_all"
