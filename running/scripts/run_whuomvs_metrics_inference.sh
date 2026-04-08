#!/bin/bash
# Run WHU-OMVS metric inference for feedforward pipeline
# Args:
#     TRAIN_TEST_SPLIT: choose train/test/predict
#     MODEL_NAME: depthanything3/mapanything/pi3/vggt
#     GPU_ID: choose an available gpu id
#     ALIGN_MODE: none/median


DATASET_PATH="/data2/dataset/Redal/work_feedforward_3drepo/dataset/WHU-OMVS"
CONFIG_PATH="/data2/dataset/Redal/work_feedforward_3drepo/configs/base_config.yaml"
TRAIN_TEST_SPLIT="test"
MODEL_NAME="vggt"
ALIGN_MODE="median"
BATCH_SIZE=8
CAME_IDS=(3)
GPU_ID=6


# run metric inference
if [[ "$TRAIN_TEST_SPLIT" == "train" ]]; then
	AREA_IDS=("area1" "area4" "area5" "area6")
	for AREA_ID in "${AREA_IDS[@]}"; do
		for CAME_ID in "${CAME_IDS[@]}"; do
			echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running metrics for ${AREA_ID} came${CAME_ID} (${MODEL_NAME})..."
			CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$(dirname "$0")/../inference/run_whuomvs_metric_inference.py" \
				--config_path "$CONFIG_PATH" \
				--dataset_path "$DATASET_PATH" \
				--split "$TRAIN_TEST_SPLIT" \
				--areas "$AREA_ID" \
				--camera_id "$CAME_ID" \
				--batch_size "$BATCH_SIZE" \
				--align_mode "$ALIGN_MODE" \
				--model_name "$MODEL_NAME" \
				--output_path "./exp/whu-omvs/metric_eval/${MODEL_NAME}_${TRAIN_TEST_SPLIT}_${AREA_ID}_came${CAME_ID}"
		done
	done
elif [[ "$TRAIN_TEST_SPLIT" == "test" ]]; then
	AREA_IDS=("area2" "area3")
	for AREA_ID in "${AREA_IDS[@]}"; do
		for CAME_ID in "${CAME_IDS[@]}"; do
			echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running metrics for ${AREA_ID} came${CAME_ID} (${MODEL_NAME})..."
			CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$(dirname "$0")/../inference/run_whuomvs_metric_inference.py" \
				--config_path "$CONFIG_PATH" \
				--dataset_path "$DATASET_PATH" \
				--split "$TRAIN_TEST_SPLIT" \
				--areas "$AREA_ID" \
				--camera_id "$CAME_ID" \
				--batch_size "$BATCH_SIZE" \
				--align_mode "$ALIGN_MODE" \
				--model_name "$MODEL_NAME" \
				--output_path "./exp/whu-omvs/metric_eval/${MODEL_NAME}_${TRAIN_TEST_SPLIT}_${AREA_ID}_came${CAME_ID}"
		done
	done
elif [[ "$TRAIN_TEST_SPLIT" == "predict" ]]; then
	# for predict folder, no area id, only camera id
	for CAME_ID in "${CAME_IDS[@]}"; do
		echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running metrics for came${CAME_ID} (${MODEL_NAME})..."
		CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$(dirname "$0")/../inference/run_whuomvs_metric_inference.py" \
			--config_path "$CONFIG_PATH" \
			--dataset_path "$DATASET_PATH" \
			--split "$TRAIN_TEST_SPLIT" \
			--camera_id "$CAME_ID" \
			--batch_size "$BATCH_SIZE" \
			--align_mode "$ALIGN_MODE" \
			--model_name "$MODEL_NAME" \
			--output_path "./exp/whu-omvs/metric_eval/${MODEL_NAME}_${TRAIN_TEST_SPLIT}_came${CAME_ID}"
	done
fi
