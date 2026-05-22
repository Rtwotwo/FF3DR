import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from running.training.datasets_adamvs import find_dataset_def
from running.utils.adamvs_utils import *
from running.training.datasets_adamvs.data_io import save_pfm, write_red_cam, read_pfm
import matplotlib.pyplot as plt

from running.metrics.dsm_metrics import (
    CameraParams, ImageParams, DSMGrid,
    load_camera_params, load_image_params, load_dsm_tif,
    build_image_name_to_params, depth_to_dsm,
    compute_elevation_error_per_pixel, compute_dsm_metrics_against_gt,
    DSMMetricAccumulator,
)
from running.metrics.accumulator import DepthMetricAccumulator

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth for whu-omvs test set')
parser.add_argument('--model', default='adamvs', help='select model from [msrednet, adamvs]')
parser.add_argument('--dataset', default='predict_oblique', help='select dataset')
parser.add_argument('--data_folder', default='/data2/dataset/Redal/work_feedforward_3drepo/dataset/WHU-OMVS/predict/source', help='test datapath')
parser.add_argument('--output_folder', default='/data2/dataset/Redal/work_feedforward_3drepo/exp/adamvs_whuomvs/MVS', help='output dir')
parser.add_argument('--loadckpt', default='/data2/dataset/Redal/work_feedforward_3drepo/weights/adamvs/model_000014_0.1409.ckpt', help='load a specific checkpoint')

# input parameters
parser.add_argument('--view_num', type=int, default=5, help='Number of images (1 ref image and view_num - 1 view images).')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--max_w', type=int, default=3712, help='Maximum image width')
parser.add_argument('--max_h', type=int, default=5504, help='Maximum image height')
parser.add_argument('--min_interval', type=float, default=0.1, help='min_interval in the bottom stage')

parser.add_argument('--fext', type=str, default='.jpg', help='Type of images.')
parser.add_argument('--normalize', type=str, default='mean', help='methods of center_image, mean[mean var] or standard[0-1].') # attention: CasMVSNet [mean var];; CasREDNet [0-255]
parser.add_argument('--resize_scale', type=float, default=0.5, help='output scale for depth and image (W and H)')
parser.add_argument('--sample_scale', type=float, default=1, help='Downsample scale for building cost volume (W and H)')
parser.add_argument('--interval_scale', type=float, default=1, help='the number of depth values')
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--display', default=True, help='display depth images')

# Cascade parameters
parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')
parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')

# split parameters
parser.add_argument('--split', type=str, default='predict', choices=['predict', 'test'], help='Dataset split: predict or test')
parser.add_argument('--areas', nargs='*', default=None, help='Area names for test split (e.g. area2 area3). Auto-detected if not specified.')
parser.add_argument('--test_max_samples_per_camera', type=int, default=20,
                    help='Max samples to keep per reference camera on test split. Set <=0 to disable.')

# metric evaluation parameters
parser.add_argument('--eval_metrics', action='store_true', help='Evaluate depth and DSM metrics after inference')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset root path for GT data. Defaults to dataset/WHU-OMVS/{split}.')
parser.add_argument('--outlier_threshold', type=float, default=20.0, help='Outlier threshold in meters for DSM MAE computation')
parser.add_argument('--align_mode', type=str, default='none', choices=['none', 'median'], help='Depth alignment mode for depth metrics (Ada-MVS outputs metric depth, so none is default)')

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)

if args.dataset_path is None:
    args.dataset_path = os.path.join(PROJECT_ROOT, 'dataset', 'WHU-OMVS', args.split)

if args.split == 'test':
    if args.areas is None:
        index_file = os.path.join(args.dataset_path, 'index.txt')
        if os.path.exists(index_file):
            with open(index_file, 'r') as fh:
                args.areas = [line.strip() for line in fh.readlines() if line.strip()]
        else:
            args.areas = sorted([d for d in os.listdir(args.dataset_path)
                                 if os.path.isdir(os.path.join(args.dataset_path, d)) and d.startswith('area')])
    args.dataset = 'test_oblique'
    if not args.data_folder or args.data_folder.endswith('/source') or args.data_folder.endswith('\\source') or args.data_folder.endswith('/predict/source'):
        args.data_folder = None


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


def evaluate_metrics(output_folder, dataset_path, split="predict", area_name=None):
    print("\n" + "=" * 60)
    print("Evaluating Ada-MVS depth and DSM metrics (split={}, area={})".format(split, area_name))
    print("=" * 60)

    if split == "test":
        return _evaluate_test_metrics(output_folder, dataset_path, area_name)

    source_dir = os.path.join(dataset_path, "source")
    camera_info_path = os.path.join(source_dir, "camera_info.txt")
    image_info_path = os.path.join(source_dir, "image_info.txt")
    dsm_path = os.path.join(dataset_path, "GT", "GT_DSM", "dsm", "0_0.tif")

    camera_params = load_camera_params(camera_info_path)
    image_params = load_image_params(image_info_path)
    name_to_params = build_image_name_to_params(image_params, camera_params)
    print("[INFO] Loaded {} image-camera param pairs".format(len(name_to_params)))

    dsm_grid = None
    if os.path.exists(dsm_path):
        dsm_grid = load_dsm_tif(dsm_path)
        print("[INFO] Loaded DSM grid: {}x{}, GSD={:.2f}m".format(
            dsm_grid.width, dsm_grid.height, dsm_grid.gsd))
    else:
        print("[WARN] DSM file not found: {}, will use per-pixel elevation error only".format(dsm_path))

    depth_accumulator = DepthMetricAccumulator(align_mode=args.align_mode)
    dsm_accumulator = DSMMetricAccumulator(outlier_threshold=args.outlier_threshold)

    cam_ids = sorted([d for d in os.listdir(output_folder)
                      if os.path.isdir(os.path.join(output_folder, d)) and d.isdigit()])

    total_frames = 0
    for cam_id in cam_ids:
        cam_dir = os.path.join(output_folder, cam_id)
        gt_depth_dir = os.path.join(dataset_path, "GT", "GT_Depths", cam_id)

        pfm_files = sorted([f for f in os.listdir(cam_dir) if f.endswith("_init.pfm")])
        for pfm_file in pfm_files:
            stem = pfm_file.replace("_init.pfm", "")
            pred_depth_path = os.path.join(cam_dir, pfm_file)
            gt_depth_path = os.path.join(gt_depth_dir, stem + ".exr")

            if not os.path.exists(gt_depth_path):
                print("[WARN] GT depth not found: {}".format(gt_depth_path))
                continue

            pred_depth, _ = read_pfm(pred_depth_path)
            gt_depth = _load_exr_depth(gt_depth_path)

            mask = (gt_depth > 1e-8) & (pred_depth > 1e-8)
            if mask.sum() == 0:
                continue

            if pred_depth.shape != gt_depth.shape:
                pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]),
                                        interpolation=cv2.INTER_LINEAR)

            depth_accumulator.update(pred_depth, gt_depth, mask)

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

                if dsm_grid is not None and total_frames < 5:
                    pred_dsm = depth_to_dsm(
                        pred_depth, cam_param, img_param, dsm_grid,
                        downsample_factor=downsample_factor,
                    )
            total_frames += 1

    depth_metrics = depth_accumulator.finalize()
    dsm_metrics = dsm_accumulator.finalize()

    dsm_output = None
    if any(v != 0.0 for v in dsm_metrics.values() if isinstance(v, (int, float))):
        dsm_output = dsm_metrics

    result = {
        "model_name": "adamvs",
        "split": "predict",
        "dataset_path": dataset_path,
        "camera_id": "all",
        "align_mode": args.align_mode,
        "areas": {"predict": depth_metrics},
        "overall": depth_metrics,
        "dsm_metrics": dsm_output,
        "total_frames": total_frames,
        "outlier_threshold": args.outlier_threshold,
    }

    print("\n" + "=" * 60)
    print("WHU-OMVS Ada-MVS Depth Metrics")
    print("=" * 60)
    print("model={} split=predict camera=all align={}".format(args.align_mode))
    headers = ["abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog", "delta1", "delta2", "delta3"]
    col_width = 12
    print(f"{'':>8} | " + " | ".join(f"{h:>{col_width}}" for h in headers))
    print("-" * (10 + len(headers) * (col_width + 3)))
    values = " | ".join(f"{depth_metrics.get(h, 0):>{col_width}.6f}" for h in headers)
    print(f"{'predict':>8} | {values}")
    print("-" * (10 + len(headers) * (col_width + 3)))
    values = " | ".join(f"{depth_metrics.get(h, 0):>{col_width}.6f}" for h in headers)
    print(f"{'overall':>8} | {values}")
    print("=" * 60)

    if dsm_grid is not None:
        print("\nAda-MVS DSM Metrics (outlier_threshold={}m)".format(args.outlier_threshold))
        print("-" * 40)
        for k, v in dsm_metrics.items():
            if isinstance(v, float):
                print("  {:20s}: {:.6f}".format(k, v))
            else:
                print("  {:20s}: {}".format(k, v))

    metrics_file = os.path.join(output_folder, "adamvs_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(result, f, indent=2)
    print("\n[INFO] Metrics saved to {}".format(metrics_file))

    return result, depth_accumulator


def _evaluate_test_metrics(output_folder, dataset_path, area_name=None):
    print("[INFO] test split: evaluating depth metrics for area={}".format(area_name))

    area_dir = os.path.join(dataset_path, area_name) if area_name else dataset_path

    depth_accumulator = DepthMetricAccumulator(align_mode=args.align_mode)

    cam_ids = sorted([d for d in os.listdir(output_folder)
                      if os.path.isdir(os.path.join(output_folder, d)) and d.isdigit()])

    total_frames = 0
    for cam_id in cam_ids:
        cam_dir = os.path.join(output_folder, cam_id)
        gt_depth_dir = os.path.join(area_dir, "depths", cam_id)

        if not os.path.isdir(gt_depth_dir):
            print("[WARN] GT depth dir not found: {}".format(gt_depth_dir))
            continue

        pfm_files = sorted([f for f in os.listdir(cam_dir) if f.endswith("_init.pfm")])
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
                    pred_depth,
                    (gt_depth.shape[1], gt_depth.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            mask_path = os.path.join(area_dir, "masks", cam_id, stem + ".png")
            if os.path.exists(mask_path):
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = (mask_img > 127) & (gt_depth > 1e-8) & (pred_depth > 1e-8)
            else:
                mask = (gt_depth > 1e-8) & (pred_depth > 1e-8)

            if mask.sum() == 0:
                continue

            depth_accumulator.update(pred_depth, gt_depth, mask)
            total_frames += 1

    depth_metrics = depth_accumulator.finalize()

    print("\n" + "=" * 60)
    print("WHU-OMVS Ada-MVS Depth Metrics [test/{}]".format(area_name))
    print("=" * 60)
    headers = ["abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog", "delta1", "delta2", "delta3"]
    col_width = 12
    print(f"{'':>8} | " + " | ".join(f"{h:>{col_width}}" for h in headers))
    print("-" * (10 + len(headers) * (col_width + 3)))
    values = " | ".join(f"{depth_metrics.get(h, 0):>{col_width}.6f}" for h in headers)
    print(f"{'area':>8} | {values}")
    print("=" * 60)
    print("[INFO] test split: DSM metrics and recon metrics skipped (no GT available)")

    per_area_result = {
        "model_name": "adamvs",
        "split": "test",
        "dataset_path": dataset_path,
        "camera_id": "all",
        "align_mode": args.align_mode,
        "areas": {area_name: depth_metrics},
        "overall": depth_metrics,
        "total_frames": total_frames,
    }
    area_metrics_file = os.path.join(output_folder, "adamvs_metrics.json")
    with open(area_metrics_file, "w") as f:
        json.dump(per_area_result, f, indent=2)
    print("[INFO] Per-area metrics saved to {}".format(area_metrics_file))

    return depth_metrics, depth_accumulator


# run MVS model to save depth maps and confidence maps
def predict_depth():
    if args.split == 'test':
        _predict_depth_test()
    else:
        _predict_depth_predict()


def _predict_depth_predict():
    MVSDataset = find_dataset_def(args.dataset)

    test_dataset = MVSDataset(args.data_folder, args.view_num, args)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=1, drop_last=False)

    model = _build_model()

    with torch.no_grad():
        output_folder = args.output_folder
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        step = 0
        first_start_time = time.time()

        for batch_idx, sample in enumerate(TestImgLoader):
            step = _process_sample(model, sample, output_folder, step)

        print("final, total_cnt = {}, total_time = {:3f}".format(step, time.time() - first_start_time))

    if args.eval_metrics:
        evaluate_metrics(output_folder, args.dataset_path, split="predict")


def _predict_depth_test():
    areas = args.areas
    if not areas:
        print("[ERROR] No areas specified for test split")
        return

    model = _build_model()
    all_area_metrics = {}
    all_area_accumulators = {}

    for area_name in areas:
        print("\n" + "=" * 60)
        print("Processing test area: {}".format(area_name))
        print("=" * 60)

        data_folder = os.path.join(args.dataset_path, area_name, "info")
        output_folder = os.path.join(args.output_folder, area_name)

        if not os.path.isdir(data_folder):
            print("[WARN] Area info dir not found: {}, skipping".format(data_folder))
            continue

        MVSDataset = find_dataset_def('test_oblique')
        test_dataset = MVSDataset(data_folder, args.view_num, args)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=1, drop_last=False)

        with torch.no_grad():
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder, exist_ok=True)

            step = 0
            first_start_time = time.time()
            test_viz_limit = 20
            original_display = args.display

            for batch_idx, sample in enumerate(TestImgLoader):
                if step < test_viz_limit:
                    args.display = original_display
                else:
                    args.display = False
                step = _process_sample(model, sample, output_folder, step)

            args.display = original_display

            print("Area {} done, total_cnt = {}, total_time = {:3f}".format(area_name, step, time.time() - first_start_time))

        if args.eval_metrics:
            area_metrics, area_accum = evaluate_metrics(output_folder, args.dataset_path, split="test", area_name=area_name)
            all_area_metrics[area_name] = area_metrics
            all_area_accumulators[area_name] = area_accum

    if args.eval_metrics and all_area_metrics:
        _save_test_summary(all_area_metrics, all_area_accumulators)


def _build_model():
    model = None
    if args.model == 'msrednet':
        from models.adamvs.msrednet import Infer_CascadeREDNet
        model = Infer_CascadeREDNet(num_depth=args.numdepth, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                              depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                              share_cr=args.share_cr,
                              cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch])

    elif args.model == 'adamvs':
        from models.adamvs.adamvs import Infer_AdaMVSNet
        model = Infer_AdaMVSNet(num_depth=args.numdepth, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                               depth_intervals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                               share_cr=args.share_cr,
                               cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch])
    else:
        raise Exception("{}? Not implemented yet!".format(args.model))

    model = nn.DataParallel(model)
    model.cuda()

    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
    model.eval()
    return model


def _process_sample(model, sample, output_folder, step):
    start_time = time.time()
    sample_cuda = tocuda(sample)

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    outputs = tensor2numpy(outputs)
    depth_est = outputs["depth"]
    photometric_confidence = outputs["photometric_confidence"]
    duration = time.time()

    depth_est = np.float32(np.squeeze(tensor2numpy(depth_est)))
    prob = np.float32(np.squeeze(tensor2numpy(photometric_confidence)))
    ref_image = np.squeeze(tensor2numpy(sample["outimage"]))
    ref_cam = np.squeeze(tensor2numpy(sample["outcam"]))
    vid = sample["out_view"][0]
    name = sample["out_name"][0]
    ref_path = np.squeeze(sample["ref_image_path"])

    output_folder2 = output_folder + ('/%s/' % vid)
    if not os.path.exists(output_folder2 + '/color/'):
        os.makedirs(output_folder2, exist_ok=True)
        os.makedirs(output_folder2 + '/color/', exist_ok=True)

    init_depth_map_path = output_folder2 + ('/%s_init.pfm' % name)
    prob_map_path = output_folder2 + ('/%s_prob.pfm' % name)
    out_ref_image_path = output_folder2 + ('/%s.jpg' % name)
    out_ref_cam_path = output_folder2 + ('/%s.txt' % name)

    if args.display:
        size1 = len(depth_est)
        size2 = len(depth_est[1])
        e = np.ones((size1, size2), dtype=np.float32)
        out_init_depth_image = e * 36000 - depth_est
        color_depth_map_path = output_folder2 + ('/color/%s_init.png' % name)
        color_prob_map_path = output_folder2 + ('/color/%s_prob.png' % name)

        for i in range(out_init_depth_image.shape[1]):
            col = out_init_depth_image[:, i]
            col[np.isinf(col)] = np.nan
            col[np.isnan(col)] = np.nanmin(col) - 1
            out_init_depth_image[:, i] = col

        plt.imsave(color_depth_map_path, out_init_depth_image, format='png')
        plt.imsave(color_prob_map_path,  np.nan_to_num(prob).clip(0, 1), format='png')

    save_pfm(init_depth_map_path, depth_est)
    save_pfm(prob_map_path, prob)
    plt.imsave(out_ref_image_path, ref_image, format='png')
    write_red_cam(out_ref_cam_path, ref_cam, ref_path)

    del outputs, sample_cuda

    step = step + 1
    save_tesult_time = time.time()
    print('depth inference {} finished, image {} finished, ({:3f}s and {:3f} sec/step)'.format(step, name, duration-start_time, save_tesult_time-duration))
    return step


def _save_test_summary(all_area_metrics, all_area_accumulators):
    areas_json = {}
    overall_accum = DepthMetricAccumulator(align_mode=args.align_mode)

    for area_name in sorted(all_area_metrics.keys()):
        areas_json[area_name] = all_area_metrics[area_name]
        if area_name in all_area_accumulators:
            overall_accum.merge(all_area_accumulators[area_name])

    overall_metrics = overall_accum.finalize()

    result = {
        "model_name": "adamvs",
        "split": "test",
        "dataset_path": args.dataset_path,
        "camera_id": "all",
        "align_mode": args.align_mode,
        "areas": areas_json,
        "overall": overall_metrics,
    }

    area_names_str = "_".join(sorted(all_area_metrics.keys()))
    metrics_filename = "test_{}_all_adamvs_metrics.json".format(area_names_str)
    metrics_file = os.path.join(args.output_folder, metrics_filename)
    with open(metrics_file, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print("WHU-OMVS Ada-MVS Depth Metrics (test split)")
    print("=" * 60)
    headers = ["abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog", "delta1", "delta2", "delta3"]
    col_width = 12
    print(f"{'':>8} | " + " | ".join(f"{h:>{col_width}}" for h in headers))
    print("-" * (10 + len(headers) * (col_width + 3)))
    for area_name in sorted(all_area_metrics.keys()):
        area_metrics = all_area_metrics[area_name]
        values = " | ".join(f"{area_metrics.get(h, 0):>{col_width}.6f}" for h in headers)
        print(f"{area_name:>8} | {values}")
    print("-" * (10 + len(headers) * (col_width + 3)))
    values = " | ".join(f"{overall_metrics.get(h, 0):>{col_width}.6f}" for h in headers)
    print(f"{'overall':>8} | {values}")
    print("=" * 60)

    print("\n[INFO] Test summary saved to {}".format(metrics_file))


if __name__ == '__main__':
    # step1. save all the depth maps and the masks in outputs directory
    predict_depth()
