#!/usr/bin/env python3
"""Evaluate Ada-MVS depth and DSM metrics on WHU-OMVS predict split."""
import json
import math
import os
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from running.metrics.dsm_metrics import (
    load_camera_params, load_image_params, load_dsm_tif,
    build_image_name_to_params, depth_to_dsm,
    compute_elevation_error_per_pixel,
    DSMMetricAccumulator,
)
from running.training.datasets_adamvs.data_io import read_pfm


def _load_exr_depth(exr_path):
    import OpenEXR, Imath
    exr_file = OpenEXR.InputFile(str(exr_path))
    header = exr_file.header()
    dw = header['dataWindow']
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_str = exr_file.channel('Y', pt)
    depth = np.frombuffer(depth_str, dtype=np.float32).reshape(h, w)
    return depth


def _compute_depth_metrics_fast(pred_depth, gt_depth, mask):
    pred = pred_depth[mask].astype(np.float64)
    gt = gt_depth[mask].astype(np.float64)

    abs_rel = np.mean(np.abs(pred - gt) / gt)
    sq_rel = np.mean(np.square(pred - gt) / gt)
    rmse = math.sqrt(np.mean(np.square(pred - gt)))
    ratio = np.maximum(pred / gt, gt / pred)
    delta1 = np.mean(ratio < 1.25)
    delta2 = np.mean(ratio < 1.25 ** 2)
    delta3 = np.mean(ratio < 1.25 ** 3)

    return {
        "abs_rel": float(abs_rel),
        "sq_rel": float(sq_rel),
        "rmse": float(rmse),
        "delta1": float(delta1),
        "delta2": float(delta2),
        "delta3": float(delta3),
        "valid_count": int(mask.sum()),
    }


def main():
    output_folder = '/data2/dataset/Redal/work_feedforward_3drepo/exp/adamvs_whuomvs/MVS'
    dataset_path = '/data2/dataset/Redal/work_feedforward_3drepo/dataset/WHU-OMVS/predict'
    outlier_threshold = 20.0
    ds_for_depth_metrics = 4

    print("=" * 60)
    print("Evaluating Ada-MVS depth and DSM metrics")
    print("=" * 60)

    source_dir = os.path.join(dataset_path, "source")
    camera_params = load_camera_params(os.path.join(source_dir, "camera_info.txt"))
    image_params = load_image_params(os.path.join(source_dir, "image_info.txt"))
    name_to_params = build_image_name_to_params(image_params, camera_params)
    print("[INFO] Loaded {} image-camera param pairs".format(len(name_to_params)))

    dsm_path = os.path.join(dataset_path, "GT", "GT_DSM", "dsm", "0_0.tif")
    dsm_grid = None
    if os.path.exists(dsm_path):
        from pathlib import Path
        dsm_grid = load_dsm_tif(Path(dsm_path))
        print("[INFO] Loaded DSM grid: {}x{}, GSD={:.2f}m".format(
            dsm_grid.width, dsm_grid.height, dsm_grid.gsd))

    dsm_accumulator = DSMMetricAccumulator(outlier_threshold=outlier_threshold)

    all_abs_rel = []
    all_sq_rel = []
    all_rmse = []
    all_delta1 = []
    all_delta2 = []
    all_delta3 = []
    all_valid = 0

    cam_ids = sorted([d for d in os.listdir(output_folder)
                      if os.path.isdir(os.path.join(output_folder, d)) and d.isdigit()])
    print("[INFO] Camera IDs: {}".format(cam_ids))

    total_frames = 0
    t0 = time.time()
    for cam_id in cam_ids:
        cam_dir = os.path.join(output_folder, cam_id)
        gt_depth_dir = os.path.join(dataset_path, "GT", "GT_Depths", cam_id)

        pfm_files = sorted([f for f in os.listdir(cam_dir) if f.endswith("_init.pfm")])
        print("[INFO] Cam {}: {} depth maps".format(cam_id, len(pfm_files)))

        for pfm_file in pfm_files:
            stem = pfm_file.replace("_init.pfm", "")
            pred_depth_path = os.path.join(cam_dir, pfm_file)
            gt_depth_path = os.path.join(gt_depth_dir, stem + ".exr")

            if not os.path.exists(gt_depth_path):
                continue

            pred_depth, _ = read_pfm(pred_depth_path)
            gt_depth = _load_exr_depth(gt_depth_path)

            if pred_depth.shape != gt_depth.shape:
                pred_depth = cv2.resize(
                    pred_depth, (gt_depth.shape[1], gt_depth.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            mask = (gt_depth > 1e-8) & (pred_depth > 1e-8)
            if mask.sum() == 0:
                continue

            if ds_for_depth_metrics > 1:
                small_h = gt_depth.shape[0] // ds_for_depth_metrics
                small_w = gt_depth.shape[1] // ds_for_depth_metrics
                pred_small = cv2.resize(pred_depth, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                gt_small = cv2.resize(gt_depth, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                mask_small = (gt_small > 1e-8) & (pred_small > 1e-8)
            else:
                pred_small = pred_depth
                gt_small = gt_depth
                mask_small = mask

            m = _compute_depth_metrics_fast(pred_small, gt_small, mask_small)
            n = m["valid_count"]
            all_abs_rel.append(m["abs_rel"] * n)
            all_sq_rel.append(m["sq_rel"] * n)
            all_rmse.append(m["rmse"] ** 2 * n)
            all_delta1.append(m["delta1"] * n)
            all_delta2.append(m["delta2"] * n)
            all_delta3.append(m["delta3"] * n)
            all_valid += n

            image_name = "{}/{}.png".format(cam_id, stem)
            param_pair = name_to_params.get(image_name)
            if param_pair is not None:
                cam_param, img_param = param_pair
                downsample_factor = cam_param.width / gt_depth.shape[1]
                elevation_error, valid = compute_elevation_error_per_pixel(
                    pred_depth, gt_depth, cam_param, img_param,
                    downsample_factor=downsample_factor,
                )
                dsm_accumulator.update(elevation_error, valid)

            total_frames += 1
            if total_frames % 10 == 0:
                elapsed = time.time() - t0
                print("[INFO] Processed {} frames, {:.1f}s elapsed".format(total_frames, elapsed))

    print("[INFO] Total frames: {}, total time: {:.1f}s".format(total_frames, time.time() - t0))

    depth_metrics = {
        "abs_rel": sum(all_abs_rel) / all_valid if all_valid > 0 else float('nan'),
        "sq_rel": sum(all_sq_rel) / all_valid if all_valid > 0 else float('nan'),
        "rmse": math.sqrt(sum(all_rmse) / all_valid) if all_valid > 0 else float('nan'),
        "delta1": sum(all_delta1) / all_valid if all_valid > 0 else float('nan'),
        "delta2": sum(all_delta2) / all_valid if all_valid > 0 else float('nan'),
        "delta3": sum(all_delta3) / all_valid if all_valid > 0 else float('nan'),
        "valid_count": all_valid,
    }
    dsm_metrics = dsm_accumulator.finalize()

    print()
    print("=" * 60)
    print("Ada-MVS Depth Metrics (align_mode=none, ds={})".format(ds_for_depth_metrics))
    print("=" * 60)
    for k, v in depth_metrics.items():
        if isinstance(v, float):
            print("  {:20s}: {:.6f}".format(k, v))
        else:
            print("  {:20s}: {}".format(k, v))

    print()
    print("Ada-MVS DSM Metrics (outlier_threshold={}m)".format(outlier_threshold))
    print("-" * 40)
    for k, v in dsm_metrics.items():
        if isinstance(v, float):
            print("  {:20s}: {:.6f}".format(k, v))
        else:
            print("  {:20s}: {}".format(k, v))

    results = {
        "model": "Ada-MVS",
        "depth_metrics": depth_metrics,
        "dsm_metrics": dsm_metrics,
        "total_frames": total_frames,
        "outlier_threshold": outlier_threshold,
        "align_mode": "none",
    }
    metrics_file = os.path.join(output_folder, "adamvs_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print("[INFO] Metrics saved to {}".format(metrics_file))


if __name__ == "__main__":
    main()
