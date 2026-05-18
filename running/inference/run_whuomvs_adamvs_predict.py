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

# metric evaluation parameters
parser.add_argument('--eval_metrics', action='store_true', help='Evaluate depth and DSM metrics after inference')
parser.add_argument('--dataset_path', type=str, default='/data2/dataset/Redal/work_feedforward_3drepo/dataset/WHU-OMVS/predict', help='Dataset root path for GT data')
parser.add_argument('--outlier_threshold', type=float, default=20.0, help='Outlier threshold in meters for DSM MAE computation')
parser.add_argument('--align_mode', type=str, default='none', choices=['none', 'median'], help='Depth alignment mode for depth metrics (Ada-MVS outputs metric depth, so none is default)')

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)


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


def evaluate_metrics(output_folder, dataset_path):
    print("\n" + "=" * 60)
    print("Evaluating Ada-MVS depth and DSM metrics")
    print("=" * 60)

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

    results = {
        "model": "Ada-MVS",
        "depth_metrics": depth_metrics,
        "dsm_metrics": dsm_metrics,
        "total_frames": total_frames,
        "outlier_threshold": args.outlier_threshold,
        "align_mode": args.align_mode,
    }

    print("\n" + "=" * 60)
    print("Ada-MVS Depth Metrics (align_mode={})".format(args.align_mode))
    print("=" * 60)
    for k, v in depth_metrics.items():
        if isinstance(v, float):
            print("  {:20s}: {:.6f}".format(k, v))
        else:
            print("  {:20s}: {}".format(k, v))

    print("\nAda-MVS DSM Metrics (outlier_threshold={}m)".format(args.outlier_threshold))
    print("-" * 40)
    for k, v in dsm_metrics.items():
        if isinstance(v, float):
            print("  {:20s}: {:.6f}".format(k, v))
        else:
            print("  {:20s}: {}".format(k, v))

    metrics_file = os.path.join(output_folder, "adamvs_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)
    print("\n[INFO] Metrics saved to {}".format(metrics_file))

    return results


# run MVS model to save depth maps and confidence maps
def predict_depth():
    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)

    test_dataset = MVSDataset(args.data_folder, args.view_num, args)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=1, drop_last=False)

    # build model
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

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
    model.eval()

    with torch.no_grad():
        # create output folder
        output_folder = args.output_folder
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        step = 0
        first_start_time = time.time()

        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            sample_cuda = tocuda(sample)

            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            outputs = tensor2numpy(outputs)
            depth_est = outputs["depth"]
            photometric_confidence = outputs["photometric_confidence"]
            duration = time.time()

            # save results
            depth_est = np.float32(np.squeeze(tensor2numpy(depth_est)))
            prob = np.float32(np.squeeze(tensor2numpy(photometric_confidence)))
            ref_image = np.squeeze(tensor2numpy(sample["outimage"]))
            ref_cam = np.squeeze(tensor2numpy(sample["outcam"]))
            #  aerial dataset
            vid = sample["out_view"][0]
            name = sample["out_name"][0]
            ref_path = np.squeeze(sample["ref_image_path"])

            # paths
            output_folder2 = output_folder + ('/%s/' % vid)
            if not os.path.exists(output_folder2 + '/color/'):
                os.mkdir(output_folder2)
                os.mkdir(output_folder2 + '/color/')

            init_depth_map_path = output_folder2 + ('/%s_init.pfm' % name)
            prob_map_path = output_folder2 + ('/%s_prob.pfm' % name)
            out_ref_image_path = output_folder2 + ('/%s.jpg' % name)
            out_ref_cam_path = output_folder2 + ('/%s.txt' % name)

            if args.display:
                # color output
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

        print("final, total_cnt = {}, total_time = {:3f}".format(step, time.time() - first_start_time))

    if args.eval_metrics:
        evaluate_metrics(output_folder, args.dataset_path)


if __name__ == '__main__':
    # step1. save all the depth maps and the masks in outputs directory
    predict_depth()
