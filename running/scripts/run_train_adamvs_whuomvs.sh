#!/bin/bash
# Train Ada-MVS on WHU-OMVS dataset
# Ada-MVS: Adaptive Multi-View Stereo (Liu et al. 2023)
#
# Usage:
#   bash running/scripts/train_adamvs_whuomvs.sh
#   bash running/scripts/train_adamvs_whuomvs.sh --gpu_id 0 --epochs 80
#   bash running/scripts/train_adamvs_whuomvs.sh --resume --logdir ./exp/adamvs_whuomvs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="adamvs"
DATASET="cas_total_rscv"
SET_NAME="whu_omvs"
TRAIN_PATH="${PROJECT_ROOT}/dataset/WHU-OMVS/train"
TEST_PATH="${PROJECT_ROOT}/dataset/WHU-OMVS/test"
LOGDIR="${PROJECT_ROOT}/exp/adamvs_whuomvs"
LOADCKPT="${PROJECT_ROOT}/weights/adamvs/model_000014_0.1409.ckpt"

VIEW_NUM=5
INTERVAL_SCALE=1
NDEPTHS="48,32,8"
MIN_INTERVAL=0.1
DEPTH_INTER_R="4,2,1"
DLOSSW="0.5,1.0,2.0"
CR_BASE_CHS="8,8,8"

EPOCHS=80
LR=0.001
LREPOCHS="10,12,14:2"
WD=0.0
BATCH_SIZE=1
SEED=1
SAVE_FREQ=1
SUMMARY_FREQ=50
GPU_ID=0

MODE="train"
RESUME=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)           shift; MODEL="$1" ;;
        --dataset)         shift; DATASET="$1" ;;
        --set_name)        shift; SET_NAME="$1" ;;
        --trainpath)       shift; TRAIN_PATH="$1" ;;
        --testpath)        shift; TEST_PATH="$1" ;;
        --logdir)          shift; LOGDIR="$1" ;;
        --loadckpt)        shift; LOADCKPT="$1" ;;
        --view_num)        shift; VIEW_NUM="$1" ;;
        --interval_scale)  shift; INTERVAL_SCALE="$1" ;;
        --ndepths)         shift; NDEPTHS="$1" ;;
        --min_interval)    shift; MIN_INTERVAL="$1" ;;
        --depth_inter_r)   shift; DEPTH_INTER_R="$1" ;;
        --dlossw)          shift; DLOSSW="$1" ;;
        --cr_base_chs)     shift; CR_BASE_CHS="$1" ;;
        --epochs)          shift; EPOCHS="$1" ;;
        --lr)              shift; LR="$1" ;;
        --lrepochs)        shift; LREPOCHS="$1" ;;
        --wd)              shift; WD="$1" ;;
        --batch_size)      shift; BATCH_SIZE="$1" ;;
        --seed)            shift; SEED="$1" ;;
        --save_freq)       shift; SAVE_FREQ="$1" ;;
        --summary_freq)    shift; SUMMARY_FREQ="$1" ;;
        --gpu_id)          shift; GPU_ID="$1" ;;
        --mode)            shift; MODE="$1" ;;
        --resume)          RESUME=true ;;
        *)                 echo "[WARN] Unknown argument: $1" ;;
    esac
    shift
done

export CUDA_VISIBLE_DEVICES=${GPU_ID}

RESUME_FLAG=""
if [ "${RESUME}" = "true" ]; then
    RESUME_FLAG="--resume"
fi

echo "============================================"
echo "Ada-MVS Training on WHU-OMVS"
echo "============================================"
echo "  Model:          ${MODEL}"
echo "  Mode:           ${MODE}"
echo "  Train path:     ${TRAIN_PATH}"
echo "  Test path:      ${TEST_PATH}"
echo "  Log dir:        ${LOGDIR}"
echo "  Checkpoint:     ${LOADCKPT}"
echo "  GPU:            ${GPU_ID}"
echo "  Epochs:         ${EPOCHS}"
echo "  LR:             ${LR}"
echo "  Resume:         ${RESUME}"
echo "============================================"

cd "${PROJECT_ROOT}"

python3 running/training/train_adamvs_whuomvs.py \
    --mode ${MODE} \
    --model ${MODEL} \
    --set_name ${SET_NAME} \
    --dataset ${DATASET} \
    --trainpath ${TRAIN_PATH} \
    --testpath ${TEST_PATH} \
    --logdir ${LOGDIR} \
    --loadckpt ${LOADCKPT} \
    --view_num ${VIEW_NUM} \
    --interval_scale ${INTERVAL_SCALE} \
    --ndepths ${NDEPTHS} \
    --min_interval ${MIN_INTERVAL} \
    --depth_inter_r ${DEPTH_INTER_R} \
    --dlossw ${DLOSSW} \
    --cr_base_chs ${CR_BASE_CHS} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --lrepochs ${LREPOCHS} \
    --wd ${WD} \
    --batch_size ${BATCH_SIZE} \
    --seed ${SEED} \
    --save_freq ${SAVE_FREQ} \
    --summary_freq ${SUMMARY_FREQ} \
    ${RESUME_FLAG}

echo ""
echo
