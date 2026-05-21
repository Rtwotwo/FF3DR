#!/bin/bash
# Run WHU-OMVS metric inference for feedforward pipeline
# Args:
#     TRAIN_TEST_SPLIT: choose train/test/predict
#     MODEL_NAME: depthanything3/mapanything/pi3/vggt
#     GPU_ID: choose an available gpu id
#     ALIGN_MODE: none/median
#     EVAL_MODE: combined (default) - run all areas together, output per-area + overall
#                separate          - run each area individually
#                both              - run both combined and separate


DATASET_PATH="/data2/dataset/Redal/work_feedforward_3drepo/dataset/WHU-OMVS"
CONFIG_PATH="/data2/dataset/Redal/work_feedforward_3drepo/configs/base_config.yaml"
TRAIN_TEST_SPLIT="test"
MODEL_NAMES=("pi3" "vggt")
# MODEL_NAMES=("depthanything3" "mapanything" "pi3" "vggt")
ALIGN_MODE="median"
BATCH_SIZE=8
EVAL_MODE="combined"
CAME_IDS=(1 2 3 4 5)
GPU_ID=6

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFERENCE_SCRIPT="${SCRIPT_DIR}/../../inference/run_whuomvs_metric_inference.py"

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
	echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running metrics for ${MODEL_NAME}..."

	if [[ "$TRAIN_TEST_SPLIT" == "train" ]]; then
		AREA_IDS=("area1" "area4" "area5" "area6")
		for CAME_ID in "${CAME_IDS[@]}"; do
			if [[ "$EVAL_MODE" == "combined" || "$EVAL_MODE" == "both" ]]; then
				echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running combined metrics for all areas came${CAME_ID} (${MODEL_NAME})..."
				CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$INFERENCE_SCRIPT" \
					--config_path "$CONFIG_PATH" \
					--dataset_path "$DATASET_PATH" \
					--split "$TRAIN_TEST_SPLIT" \
					--areas "${AREA_IDS[@]}" \
					--camera_id "$CAME_ID" \
					--batch_size "$BATCH_SIZE" \
					--align_mode "$ALIGN_MODE" \
					--model_name "$MODEL_NAME" \
					--output_path "./exp/whu-omvs/metric_eval/${MODEL_NAME}_${TRAIN_TEST_SPLIT}_all_areas_came${CAME_ID}"
			fi
			if [[ "$EVAL_MODE" == "separate" || "$EVAL_MODE" == "both" ]]; then
				for AREA_ID in "${AREA_IDS[@]}"; do
					echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running metrics for ${AREA_ID} came${CAME_ID} (${MODEL_NAME})..."
					CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$INFERENCE_SCRIPT" \
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
			fi
		done
	elif [[ "$TRAIN_TEST_SPLIT" == "test" ]]; then
		AREA_IDS=("area2" "area3")
		for CAME_ID in "${CAME_IDS[@]}"; do
			if [[ "$EVAL_MODE" == "combined" || "$EVAL_MODE" == "both" ]]; then
				echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running combined metrics for area2+area3 came${CAME_ID} (${MODEL_NAME})..."
				CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$INFERENCE_SCRIPT" \
					--config_path "$CONFIG_PATH" \
					--dataset_path "$DATASET_PATH" \
					--split "$TRAIN_TEST_SPLIT" \
					--areas "${AREA_IDS[@]}" \
					--camera_id "$CAME_ID" \
					--batch_size "$BATCH_SIZE" \
					--align_mode "$ALIGN_MODE" \
					--model_name "$MODEL_NAME" \
					--output_path "./exp/whu-omvs/metric_eval_test/${MODEL_NAME}_${TRAIN_TEST_SPLIT}_area2_area3_came${CAME_ID}"
			fi
			if [[ "$EVAL_MODE" == "separate" || "$EVAL_MODE" == "both" ]]; then
				for AREA_ID in "${AREA_IDS[@]}"; do
					echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running metrics for ${AREA_ID} came${CAME_ID} (${MODEL_NAME})..."
					CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$INFERENCE_SCRIPT" \
						--config_path "$CONFIG_PATH" \
						--dataset_path "$DATASET_PATH" \
						--split "$TRAIN_TEST_SPLIT" \
						--areas "$AREA_ID" \
						--camera_id "$CAME_ID" \
						--batch_size "$BATCH_SIZE" \
						--align_mode "$ALIGN_MODE" \
						--model_name "$MODEL_NAME" \
						--output_path "./exp/whu-omvs/metric_eval_test/${MODEL_NAME}_${TRAIN_TEST_SPLIT}_${AREA_ID}_came${CAME_ID}"
				done
			fi
		done
	elif [[ "$TRAIN_TEST_SPLIT" == "predict" ]]; then
		for CAME_ID in "${CAME_IDS[@]}"; do
			echo "[INFO $(date +"%Y-%m-%d %H:%M:%S")] Running metrics for came${CAME_ID} (${MODEL_NAME})..."
			CUDA_VISIBLE_DEVICES=$GPU_ID python3 "$INFERENCE_SCRIPT" \
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
done
