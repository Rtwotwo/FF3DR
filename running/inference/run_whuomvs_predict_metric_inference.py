from __future__ import annotations

import argparse
import json
import logging
import os

import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import OpenEXR
import Imath
import torch
from PIL import Image
from safetensors.torch import load_file


def _find_repo_root(start: Path) -> Path:
    for current in [start, *start.parents]:
        if (current / "models").is_dir() and (current / "running").is_dir() and (current / "configs").is_dir():
            return current
    return start.parent


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _find_repo_root(_SCRIPT_DIR)
_RUNNING_DIR = _SCRIPT_DIR.parent
_DA3_SRC = _REPO_ROOT / "clones" / "Depth-Anything-3" / "src"
for path in (_REPO_ROOT, _RUNNING_DIR, _DA3_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def save_pfm(filename: str, image: np.ndarray, scale: float = 1.0) -> None:
    with open(filename, "wb") as f:
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        image = np.flipud(image)
        color = image.ndim == 3 and image.shape[2] == 3
        f.write(("PF\n" if color else "Pf\n").encode("utf-8"))
        f.write("{} {}\n".format(image.shape[1], image.shape[0]).encode("utf-8"))
        endian = image.dtype.byteorder
        if endian == "<" or (endian == "=" and sys.byteorder == "little"):
            scale = -scale
        f.write(("{:.6f}\n".format(scale)).encode("utf-8"))
        image.tofile(f)


def write_adamvs_cam(
    filepath: str,
    extrinsic_4x4: np.ndarray,
    intrinsic_3x3: np.ndarray,
    depth_min: float,
    depth_max: float,
    num_depth: int = 192,
    ref_image_path: str = "",
) -> None:
    interval = (depth_max - depth_min) / num_depth if num_depth > 0 else 0.0
    with open(filepath, "w") as f:
        f.write("extrinsic: XrightYdown, [Rcw|tcw]\n")
        for i in range(4):
            row_vals = [str(extrinsic_4x4[i, j]) for j in range(4)]
            f.write(" ".join(row_vals) + " \n")
        f.write("\n")
        f.write("intrinsic\n")
        for i in range(3):
            row_vals = [str(intrinsic_3x3[i, j]) for j in range(3)]
            f.write(" ".join(row_vals) + " \n")
        f.write("\n")
        f.write("{} {} {} {}\n".format(depth_min, interval, num_depth, depth_max))
        f.write("\n")
        f.write(str(ref_image_path) + "\n")


def align_depth_median(pred_depth: np.ndarray, gt_depth: np.ndarray) -> Tuple[np.ndarray, float]:
    valid = np.isfinite(pred_depth) & np.isfinite(gt_depth) & (pred_depth > 1e-8) & (gt_depth > 1e-8)
    if not np.any(valid):
        return pred_depth, 1.0
    scale = float(np.median(gt_depth[valid])) / float(np.median(pred_depth[valid]))
    if not np.isfinite(scale) or abs(scale) < 1e-8:
        return pred_depth, 1.0
    return pred_depth * scale, scale


def align_depth_least_squares(pred_depth: np.ndarray, gt_depth: np.ndarray) -> Tuple[np.ndarray, float]:
    valid = np.isfinite(pred_depth) & np.isfinite(gt_depth) & (pred_depth > 1e-8) & (gt_depth > 1e-8)
    if valid.sum() < 10:
        return pred_depth, 1.0
    p = pred_depth[valid].astype(np.float64)
    g = gt_depth[valid].astype(np.float64)
    scale = float(np.dot(p, g) / (np.dot(p, p) + 1e-12))
    if not np.isfinite(scale) or abs(scale) < 1e-8:
        return pred_depth, 1.0
    return pred_depth * scale, scale


def align_depth_affine(pred_depth: np.ndarray, gt_depth: np.ndarray) -> Tuple[np.ndarray, float, float]:
    valid = np.isfinite(pred_depth) & np.isfinite(gt_depth) & (pred_depth > 1e-8) & (gt_depth > 1e-8)
    if valid.sum() < 10:
        return pred_depth, 1.0, 0.0
    p = pred_depth[valid].astype(np.float64)
    g = gt_depth[valid].astype(np.float64)
    A = np.column_stack([p, np.ones_like(p)])
    result = np.linalg.lstsq(A, g, rcond=None)[0]
    scale = float(result[0])
    shift = float(result[1])
    if not np.isfinite(scale) or abs(scale) < 1e-8:
        return pred_depth, 1.0, 0.0
    return pred_depth * scale + shift, scale, shift


def build_w2c_extrinsic_from_whuomvs(Rwc: np.ndarray, twc: np.ndarray) -> np.ndarray:
    O_xrightyup = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    Rwc_down = Rwc @ O_xrightyup
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = Rwc_down
    c2w[:3, 3] = twc
    w2c = np.linalg.inv(c2w)
    return w2c


def build_intrinsic_3x3(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    K = np.zeros((3, 3), dtype=np.float64)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    K[2, 2] = 1.0
    return K

from metrics import AdaMVSMetricAccumulator, DSMMetricAccumulator
from metrics.dsm_metrics import (
    load_camera_params,
    load_image_params,
    load_dsm_tif,
    compute_elevation_error_per_pixel,
    compute_dsm_metrics_against_gt,
    depth_to_dsm,
    build_image_name_to_params,
)
from models.depthanything3.api import DepthAnything3
from models.mapanything.models.mapanything import MapAnything
from models.mapanything.utils.image import load_images
from models.pi3.models.pi3 import Pi3
from models.pi3.utils.basic import load_images_as_tensor_pi_long
from models.vggt.models.vggt import VGGT
from models.vggt.utils.load_fn import load_and_preprocess_images
from models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from uniception.models.utils.transformer_blocks import Mlp
from running.utils.config_utils import load_config

logger = logging.getLogger(__name__)


def _resolve_path(path_value: str) -> str:
    if path_value is None:
        return path_value
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((_REPO_ROOT / path).resolve())


def _parse_stem_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return int(stem), 0
    parts = stem.split("_")
    if len(parts) >= 2 and all(p.isdigit() for p in parts[:2]):
        return int(parts[0]), int(parts[1])
    return stem


def _load_exr_single_channel(path: Path, channel_name: str = "Y") -> np.ndarray:
    exr_file = OpenEXR.InputFile(str(path))
    header = exr_file.header()
    data_window = header["dataWindow"]
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_bytes = exr_file.channel(channel_name, pixel_type)
    depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(height, width)
    return depth.copy()


def _load_exr_normals(path: Path) -> np.ndarray:
    exr_file = OpenEXR.InputFile(str(path))
    header = exr_file.header()
    data_window = header["dataWindow"]
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = list(header["channels"].keys())
    normal = np.zeros((height, width, 3), dtype=np.float32)
    for idx, ch in enumerate(sorted(channels)):
        ch_bytes = exr_file.channel(ch, pixel_type)
        normal[:, :, idx] = np.frombuffer(ch_bytes, dtype=np.float32).reshape(height, width)
    return normal


def _load_mask(path: Optional[Path], shape) -> np.ndarray:
    if path is None or not path.exists():
        return np.ones(shape, dtype=np.float32)
    mask = Image.open(path).convert("L")
    mask = np.asarray(mask, dtype=np.float32) / 255.0
    return (mask > 0.5).astype(np.float32)


def _compute_normal_metrics(pred_normal: np.ndarray, gt_normal: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    valid = mask > 0.5
    valid &= np.isfinite(pred_normal).all(axis=-1) if pred_normal.ndim == 3 else np.isfinite(pred_normal)
    valid &= np.isfinite(gt_normal).all(axis=-1) if gt_normal.ndim == 3 else np.isfinite(gt_normal)

    if valid.sum() == 0:
        return {"mean_ang_error": float("nan"), "median_ang_error": float("nan"), "normal_valid_count": 0}

    pred_v = pred_normal[valid].reshape(-1, 3)
    gt_v = gt_normal[valid].reshape(-1, 3)

    pred_norm = np.linalg.norm(pred_v, axis=-1, keepdims=True).clip(min=1e-8)
    gt_norm = np.linalg.norm(gt_v, axis=-1, keepdims=True).clip(min=1e-8)
    pred_v = pred_v / pred_norm
    gt_v = gt_v / gt_norm

    cos_sim = np.sum(pred_v * gt_v, axis=-1).clip(-1.0, 1.0)
    ang_error = np.arccos(cos_sim) * 180.0 / np.pi

    return {
        "mean_ang_error": float(np.mean(ang_error)),
        "median_ang_error": float(np.median(ang_error)),
        "normal_valid_count": int(valid.sum()),
    }


class PredictMetricInfer:
    """Run feedforward model inference on WHU-OMVS predict split and compute metrics.

    The predict dataset layout:
      predict/
        Images/{cam_id}/{frame_id}.png
        GT/
          GT_Depths/{cam_id}/{frame_id}.exr        (single Y channel)
          GT_Normals/{cam_id}/{frame_id}.exr        (BGR channels)
          GT_DSM/{dsm,dom}/
          GT_Mesh/
          GT_pc/
        source/{camera_info,image_info,image_path,viewpair}.txt

    Metrics computed (matching the Ada-MVS / WHU-OMVS paper benchmark, Liu et al. 2023):

    DSM-level metrics (Eqs. 7-9 in the paper, applied at depth map level):
      - MAE (m): Mean Absolute Error with outlier threshold T (default 20m = 100*GSD)
      - RMSE (m): Root Mean Square Error with outlier threshold T
      - PAG_0.2m (%): Percentage of Accurate Grids with |error| < 0.2m
      - PAG_0.4m (%): Percentage of Accurate Grids with |error| < 0.4m
      - PAG_0.6m (%): Percentage of Accurate Grids with |error| < 0.6m

    Standard depth metrics (for cross-benchmark comparison):
      - abs_rel, sq_rel, rmse_log, log10, silog, delta1, delta2, delta3

    Normal metrics (optional):
      - mean angular error, median angular error
    """

    def __init__(self, cfg, dataset_path, camera_ids, batch_size, align_mode, eval_normal, outlier_threshold, eval_dsm=True, save_adamvs_format=False, adamvs_output_path=None, depth_align_method="median", multiview_num_neighbors=0, multiview_process_res=504, multiview_ref_strategy="saddle_balanced"):
        self.cfg = cfg
        self.dataset_path = Path(dataset_path)
        self.camera_ids = [str(cid) for cid in camera_ids]
        self.batch_size = max(1, int(batch_size))
        self.align_mode = align_mode
        self.eval_normal = eval_normal
        self.outlier_threshold = float(outlier_threshold)
        self.eval_dsm = eval_dsm
        self.save_adamvs_format = save_adamvs_format
        self.adamvs_output_path = Path(adamvs_output_path) if adamvs_output_path else None
        self.depth_align_method = depth_align_method
        self.multiview_num_neighbors = int(multiview_num_neighbors)
        self.multiview_process_res = int(multiview_process_res)
        self.multiview_ref_strategy = multiview_ref_strategy
        self.model_name = self.cfg["Model"].get("name", self.cfg["Model"].get("model_name", "depthanything3"))
        if torch.cuda.is_available():
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

        self._name_to_params = None
        self._dsm_grid = None
        if self.eval_dsm or self.save_adamvs_format:
            self._load_dsm_infrastructure()

    def _cfg_get_weight(self, model_key, field, fallback_flat_key=None):
        weights = self.cfg.get("Weights", {})
        if model_key in weights and isinstance(weights[model_key], dict) and field in weights[model_key]:
            return weights[model_key][field]
        if fallback_flat_key is not None and fallback_flat_key in weights:
            return weights[fallback_flat_key]
        raise KeyError("Missing weight config for {}.{} (fallback={})".format(model_key, field, fallback_flat_key))

    def _load_model(self):
        if self.model_name == "depthanything3":
            da3_config_path = self._cfg_get_weight("depthanything3", "DA3_CONFIG", fallback_flat_key="DA3_CONFIG")
            da3_weight_path = self._cfg_get_weight("depthanything3", "DA3", fallback_flat_key="DA3")
            with open(da3_config_path, "r", encoding="utf-8") as fh:
                da3_config = json.load(fh)
            model = DepthAnything3(**da3_config)
            state_dict = load_file(da3_weight_path)
            model.load_state_dict(state_dict, strict=False)
        elif self.model_name == "mapanything":
            map_config_path = self._cfg_get_weight("mapanything", "MAP_CONFIG", fallback_flat_key="MAP_CONFIG")
            map_weight_path = self._cfg_get_weight("mapanything", "MAP", fallback_flat_key="MAP")
            with open(map_config_path, "r", encoding="utf-8") as fh:
                map_config = json.load(fh)
            if isinstance(map_config, dict):
                info_cfg = map_config.get("info_sharing_config", {})
                module_args = info_cfg.get("module_args", {})
                if module_args.get("mlp_layer", None) == "mlp":
                    module_args["mlp_layer"] = Mlp
            model = MapAnything(**map_config)
            state_dict = load_file(map_weight_path)
            model.load_state_dict(state_dict, strict=False)
        elif self.model_name == "pi3":
            _ = self._cfg_get_weight("pi3", "PI3_CONFIG", fallback_flat_key="PI3_CONFIG")
            pi3_weight_path = self._cfg_get_weight("pi3", "PI3", fallback_flat_key="PI3")
            model = Pi3()
            state_dict = load_file(pi3_weight_path)
            model.load_state_dict(state_dict, strict=False)
        elif self.model_name == "vggt":
            vggt_weight_path = self._cfg_get_weight("vggt", "VGGT", fallback_flat_key="VGGT")
            model = VGGT()
            state_dict = torch.load(vggt_weight_path, map_location=self.device)
            if isinstance(state_dict, dict):
                model.load_state_dict(state_dict, strict=False)
        else:
            raise RuntimeError("[ERROR] model_name must be one of depthanything3/mapanything/pi3/vggt")
        model.eval().to(self.device)
        logger.info("[INFO] Model loaded: %s", self.model_name)
        return model

    def _load_dsm_infrastructure(self):
        source_dir = self.dataset_path / "source"
        camera_info_path = source_dir / "camera_info.txt"
        image_info_path = source_dir / "image_info.txt"
        dsm_path = self.dataset_path / "GT" / "GT_DSM" / "dsm" / "0_0.tif"

        if not camera_info_path.exists() or not image_info_path.exists():
            logger.warning("[WARN] Camera/image info files not found, disabling DSM evaluation")
            self.eval_dsm = False
            return

        camera_params = load_camera_params(camera_info_path)
        image_params = load_image_params(image_info_path)
        self._name_to_params = build_image_name_to_params(image_params, camera_params)
        logger.info("[INFO] Loaded %d image-camera param pairs", len(self._name_to_params))

        if dsm_path.exists():
            self._dsm_grid = load_dsm_tif(dsm_path)
            logger.info(
                "[INFO] Loaded DSM grid: %dx%d, GSD=%.2fm, origin=(%.1f, %.1f)",
                self._dsm_grid.width, self._dsm_grid.height,
                self._dsm_grid.gsd, self._dsm_grid.x_origin, self._dsm_grid.y_origin,
            )
        else:
            logger.warning("[WARN] DSM file not found: %s, will use per-pixel elevation error only", dsm_path)

    def _collect_samples_for_camera(self, cam_id: str) -> List[Dict]:
        image_dir = self.dataset_path / "Images" / cam_id
        depth_dir = self.dataset_path / "GT" / "GT_Depths" / cam_id
        normal_dir = self.dataset_path / "GT" / "GT_Normals" / cam_id

        if not image_dir.is_dir():
            logger.warning("[WARN] Image dir not found: %s", image_dir)
            return []
        if not depth_dir.is_dir():
            logger.warning("[WARN] Depth dir not found: %s", depth_dir)
            return []

        image_files = {p.stem: p for p in sorted(image_dir.glob("*.png"), key=_parse_stem_key)}
        depth_files = {p.stem: p for p in sorted(depth_dir.glob("*.exr"), key=_parse_stem_key)}
        normal_files = {p.stem: p for p in sorted(normal_dir.glob("*.exr"), key=_parse_stem_key)} if normal_dir.is_dir() else {}

        common_stems = sorted(
            set(image_files) & set(depth_files),
            key=lambda s: int(s) if s.isdigit() else s,
        )

        samples = []
        for stem in common_stems:
            sample = {
                "stem": stem,
                "image_path": image_files[stem],
                "depth_path": depth_files[stem],
                "normal_path": normal_files.get(stem),
                "cam_id": cam_id,
            }
            samples.append(sample)
        return samples

    def _predict_batch(self, image_paths):
        if self.model_name == "mapanything":
            return self._predict_batch_mapanything(image_paths)
        if self.model_name == "pi3":
            return self._predict_batch_pi3(image_paths)
        if self.model_name == "vggt":
            return self._predict_batch_vggt(image_paths)
        if self.model_name == "depthanything3" and self.multiview_num_neighbors > 0:
            return self._predict_batch_da3_single(image_paths)

        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.amp.autocast("cuda", dtype=self.dtype):
                    prediction = self.model.inference([str(p) for p in image_paths])
            else:
                prediction = self.model.inference([str(p) for p in image_paths])

        depth = prediction.depth
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
        else:
            depth = np.asarray(depth)
        if depth.ndim == 4 and depth.shape[1] == 1:
            depth = np.squeeze(depth, axis=1)
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = np.squeeze(depth, axis=-1)

        conf = None
        if hasattr(prediction, "depth_conf"):
            conf = prediction.depth_conf
        elif hasattr(prediction, "conf"):
            conf = prediction.conf
        if conf is not None:
            if isinstance(conf, torch.Tensor):
                conf = conf.detach().cpu().numpy()
            else:
                conf = np.asarray(conf)
            if conf.ndim == 4 and conf.shape[0] == 1:
                conf = np.squeeze(conf, axis=0)
            if conf.ndim == 3 and conf.shape[0] == 1:
                conf = np.squeeze(conf, axis=0)
            if conf.ndim == 3 and conf.shape[-1] == 1:
                conf = np.squeeze(conf, axis=-1)
            if conf.ndim != 2:
                conf = None
        if conf is not None and depth.ndim == 3 and conf.shape != depth.shape[-2:]:
            conf = cv2.resize(conf, (depth.shape[-1], depth.shape[-2]), interpolation=cv2.INTER_LINEAR)

        return {"depth": depth, "conf": conf}

    def _predict_batch_mapanything(self, image_paths):
        depth_list = []
        conf_list = []
        for image_path in image_paths:
            views = load_images([str(image_path)])
            with torch.no_grad():
                pred_list = self.model.infer(
                    views,
                    memory_efficient_inference=False,
                    use_amp=True,
                    amp_dtype="bf16",
                    apply_mask=True,
                    mask_edges=True,
                    apply_confidence_mask=False,
                    ignore_calibration_inputs=False,
                    ignore_depth_inputs=False,
                    ignore_pose_inputs=True,
                    ignore_depth_scale_inputs=False,
                    ignore_pose_scale_inputs=True,
                )
            depth = pred_list[0]["depth_z"]
            if isinstance(depth, torch.Tensor):
                depth = depth.detach().cpu().numpy()
            depth = np.asarray(depth, dtype=np.float32)
            depth = np.squeeze(depth)
            if depth.ndim == 3:
                if depth.shape[0] == 1:
                    depth = depth[0]
                elif depth.shape[-1] == 1:
                    depth = depth[..., 0]
            if depth.ndim != 2:
                raise RuntimeError(
                    "[ERROR] Unexpected mapanything depth shape for {}: {}".format(image_path, depth.shape)
                )
            depth_list.append(depth)

            conf = pred_list[0].get("conf")
            if conf is not None:
                if isinstance(conf, torch.Tensor):
                    conf = conf.detach().cpu().numpy()
                conf = np.asarray(conf, dtype=np.float32)
                conf = np.squeeze(conf)
                if conf.ndim == 3:
                    if conf.shape[0] == 1:
                        conf = conf[0]
                    elif conf.shape[-1] == 1:
                        conf = conf[..., 0]
                if conf.ndim != 2:
                    conf = None
            conf_list.append(conf)

        depths = np.stack(depth_list, axis=0)
        confs = None
        if all(c is not None for c in conf_list):
            confs = np.stack(conf_list, axis=0)
        return {"depth": depths, "conf": confs}

    def _predict_batch_pi3(self, image_paths):
        depth_list = []
        conf_list = []
        for image_path in image_paths:
            images = load_images_as_tensor_pi_long([str(image_path)]).to(self.device)
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.amp.autocast("cuda", dtype=self.dtype):
                        res = self.model(images[None])
                else:
                    res = self.model(images[None])
            depth = res["local_points"][..., 2]
            if isinstance(depth, torch.Tensor):
                depth = depth.detach().cpu().numpy()
            depth = np.asarray(depth)
            if depth.ndim == 4 and depth.shape[0] == 1:
                depth = np.squeeze(depth, axis=0)
            if depth.ndim == 3 and depth.shape[0] == 1:
                depth = np.squeeze(depth, axis=0)
            depth_list.append(depth)

            conf = res.get("conf")
            if conf is not None:
                conf = torch.sigmoid(conf[..., 0])
                if isinstance(conf, torch.Tensor):
                    conf = conf.detach().cpu().numpy()
                conf = np.asarray(conf)
                if conf.ndim == 4 and conf.shape[0] == 1:
                    conf = np.squeeze(conf, axis=0)
                if conf.ndim == 3 and conf.shape[0] == 1:
                    conf = np.squeeze(conf, axis=0)
                if conf.ndim != 2:
                    conf = None
            conf_list.append(conf)

        depths = np.stack(depth_list, axis=0)
        confs = None
        if all(c is not None for c in conf_list):
            confs = np.stack(conf_list, axis=0)
        return {"depth": depths, "conf": confs}

    def _predict_batch_vggt(self, image_paths):
        depth_list = []
        conf_list = []
        for image_path in image_paths:
            images = load_and_preprocess_images([str(image_path)]).to(self.device)
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.amp.autocast("cuda", dtype=self.dtype):
                        pred = self.model(images)
                else:
                    pred = self.model(images)
            pose_enc = pred["pose_enc"]
            extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            depth = pred["depth"][..., 0]
            if isinstance(depth, torch.Tensor):
                depth = depth.detach().cpu().numpy()
            depth = np.asarray(depth)
            if depth.ndim == 3 and depth.shape[0] == 1:
                depth = np.squeeze(depth, axis=0)
            depth_list.append(depth)

            conf = pred.get("depth_conf", pred.get("world_points_conf"))
            if conf is not None:
                if isinstance(conf, torch.Tensor):
                    conf = conf.detach().cpu().numpy()
                conf = np.asarray(conf)
                if conf.ndim == 3 and conf.shape[0] == 1:
                    conf = np.squeeze(conf, axis=0)
                if conf.ndim != 2:
                    conf = None
            conf_list.append(conf)

        depths = np.stack(depth_list, axis=0)
        confs = None
        if all(c is not None for c in conf_list):
            confs = np.stack(conf_list, axis=0)
        return {"depth": depths, "conf": confs}

    def _predict_batch_da3_single(self, image_paths):
        depth_list = []
        conf_list = []
        for image_path in image_paths:
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.amp.autocast("cuda", dtype=self.dtype):
                        prediction = self.model.inference([str(image_path)])
                else:
                    prediction = self.model.inference([str(image_path)])
            depth = prediction.depth
            if isinstance(depth, torch.Tensor):
                depth = depth.detach().cpu().numpy()
            else:
                depth = np.asarray(depth)
            if depth.ndim == 3 and depth.shape[0] == 1:
                depth = np.squeeze(depth, axis=0)
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = np.squeeze(depth, axis=-1)
            depth_list.append(depth)
            conf = None
            if hasattr(prediction, "depth_conf"):
                conf = prediction.depth_conf
            elif hasattr(prediction, "conf"):
                conf = prediction.conf
            if conf is not None:
                if isinstance(conf, torch.Tensor):
                    conf = conf.detach().cpu().numpy()
                else:
                    conf = np.asarray(conf)
                if conf.ndim == 3 and conf.shape[0] == 1:
                    conf = np.squeeze(conf, axis=0)
                if conf.ndim == 3 and conf.shape[-1] == 1:
                    conf = np.squeeze(conf, axis=-1)
                if conf.ndim != 2:
                    conf = None
            conf_list.append(conf)
        depths = np.stack(depth_list, axis=0)
        confs = None
        if all(c is not None for c in conf_list):
            confs = np.stack(conf_list, axis=0)
        return {"depth": depths, "conf": confs}

    def _predict_da3_multiview(self, target_samples, all_samples_for_cam):
        if self._name_to_params is None:
            logger.warning("[WARN] No camera params loaded, falling back to single-view DA3")
            return self._predict_batch_da3_single([s["image_path"] for s in target_samples])

        depth_list = []
        conf_list = []

        n_neighbors = self.multiview_num_neighbors
        half_n = n_neighbors // 2

        sample_idx_map = {s["stem"]: idx for idx, s in enumerate(all_samples_for_cam)}

        for target_sample in target_samples:
            target_stem = target_sample["stem"]
            target_idx = sample_idx_map.get(target_stem)
            if target_idx is None:
                logger.warning("[WARN] Target stem %s not found in camera samples", target_stem)
                pred = self._predict_batch_da3_single([target_sample["image_path"]])
                depth_list.append(pred["depth"][0])
                conf_list.append(pred["conf"][0] if pred["conf"] is not None else None)
                continue

            start_idx = max(0, target_idx - half_n)
            end_idx = min(len(all_samples_for_cam), target_idx + half_n + 1)
            neighbor_samples = all_samples_for_cam[start_idx:end_idx]

            target_local_idx = target_idx - start_idx

            image_paths_list = [str(s["image_path"]) for s in neighbor_samples]
            extrinsics_list = []
            intrinsics_list = []

            for s in neighbor_samples:
                image_name = "{}/{}.png".format(s["cam_id"], s["stem"])
                param_pair = self._name_to_params.get(image_name)
                if param_pair is None:
                    logger.warning("[WARN] No camera params for %s, falling back to single-view", image_name)
                    pred = self._predict_batch_da3_single([target_sample["image_path"]])
                    depth_list.append(pred["depth"][0])
                    conf_list.append(pred["conf"][0] if pred["conf"] is not None else None)
                    break
                cam_param, img_param = param_pair
                w2c = build_w2c_extrinsic_from_whuomvs(img_param.Rwc, img_param.twc)
                K = build_intrinsic_3x3(cam_param.fx, cam_param.fy, cam_param.cx, cam_param.cy)
                extrinsics_list.append(w2c.astype(np.float32))
                intrinsics_list.append(K.astype(np.float32))
            else:
                extrinsics_arr = np.stack(extrinsics_list, axis=0)
                intrinsics_arr = np.stack(intrinsics_list, axis=0)

                logger.info(
                    "[INFO] DA3 multi-view: target=%s, %d views (target at index %d)",
                    target_stem, len(neighbor_samples), target_local_idx,
                )

                with torch.no_grad():
                    if torch.cuda.is_available():
                        with torch.amp.autocast("cuda", dtype=self.dtype):
                            prediction = self.model.inference(
                                image=image_paths_list,
                                extrinsics=extrinsics_arr,
                                intrinsics=intrinsics_arr,
                                process_res=self.multiview_process_res,
                                ref_view_strategy=self.multiview_ref_strategy,
                            )
                    else:
                        prediction = self.model.inference(
                            image=image_paths_list,
                            extrinsics=extrinsics_arr,
                            intrinsics=intrinsics_arr,
                            process_res=self.multiview_process_res,
                            ref_view_strategy=self.multiview_ref_strategy,
                        )

                depth = prediction.depth
                if isinstance(depth, torch.Tensor):
                    depth = depth.detach().cpu().numpy()
                else:
                    depth = np.asarray(depth)

                target_depth = depth[target_local_idx]
                if target_depth.ndim == 3 and target_depth.shape[0] == 1:
                    target_depth = np.squeeze(target_depth, axis=0)
                if target_depth.ndim == 3 and target_depth.shape[-1] == 1:
                    target_depth = np.squeeze(target_depth, axis=-1)
                depth_list.append(target_depth)

                conf = None
                if hasattr(prediction, "depth_conf") and prediction.depth_conf is not None:
                    conf = prediction.depth_conf
                    if isinstance(conf, torch.Tensor):
                        conf = conf.detach().cpu().numpy()
                    else:
                        conf = np.asarray(conf)
                    target_conf = conf[target_local_idx]
                    if target_conf.ndim == 3 and target_conf.shape[0] == 1:
                        target_conf = np.squeeze(target_conf, axis=0)
                    if target_conf.ndim == 3 and target_conf.shape[-1] == 1:
                        target_conf = np.squeeze(target_conf, axis=-1)
                    conf = target_conf if target_conf.ndim == 2 else None
                conf_list.append(conf)

                continue

        depths = np.stack(depth_list, axis=0)
        confs = None
        if all(c is not None for c in conf_list):
            confs = np.stack(conf_list, axis=0)
        return {"depth": depths, "conf": confs}

    def _prepare_prediction(self, pred_depth, gt_depth):
        pred_depth = np.asarray(pred_depth, dtype=np.float32)
        gt_depth = np.asarray(gt_depth, dtype=np.float32)

        pred_depth = np.squeeze(pred_depth)
        if pred_depth.ndim == 3:
            if pred_depth.shape[0] == 1:
                pred_depth = pred_depth[0]
            elif pred_depth.shape[-1] == 1:
                pred_depth = pred_depth[..., 0]

        if pred_depth.ndim != 2:
            raise RuntimeError(
                "[ERROR] pred_depth must be 2D after squeeze, got shape={}, gt_shape={}".format(
                    pred_depth.shape, gt_depth.shape
                )
            )
        if gt_depth.ndim != 2 or gt_depth.shape[0] <= 0 or gt_depth.shape[1] <= 0:
            raise RuntimeError("[ERROR] Invalid gt_depth shape: {}".format(gt_depth.shape))

        if pred_depth.shape != gt_depth.shape:
            pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        return pred_depth

    def run(self):
        per_camera_results = {}
        overall_accumulator = AdaMVSMetricAccumulator(
            outlier_threshold=self.outlier_threshold,
            align_mode=self.align_mode,
        )
        overall_dsm_accumulator = DSMMetricAccumulator(
            outlier_threshold=self.outlier_threshold,
        ) if self.eval_dsm else None
        overall_normal_metrics = {"mean_ang_error": [], "median_ang_error": [], "normal_valid_count": 0}

        pred_dsm_sum = None
        pred_dsm_count = None
        gt_dsm_sum = None
        gt_dsm_count = None

        for cam_id in self.camera_ids:
            samples = self._collect_samples_for_camera(cam_id)
            if not samples:
                logger.warning("[WARN] No samples found for camera %s", cam_id)
                continue

            cam_accumulator = AdaMVSMetricAccumulator(
                outlier_threshold=self.outlier_threshold,
                align_mode=self.align_mode,
            )
            cam_dsm_accumulator = DSMMetricAccumulator(
                outlier_threshold=self.outlier_threshold,
            ) if self.eval_dsm else None
            cam_normal_errors = {"mean_ang_error": [], "median_ang_error": [], "normal_valid_count": 0}

            cam_pred_dsm_sum = None
            cam_pred_dsm_count = None
            cam_gt_dsm_sum = None
            cam_gt_dsm_count = None

            logger.info(
                "[INFO] Evaluating camera=%s samples=%d model=%s dsm=%s",
                cam_id,
                len(samples),
                self.model_name,
                self.eval_dsm,
            )

            for start_idx in range(0, len(samples), self.batch_size):
                batch_samples = samples[start_idx : start_idx + self.batch_size]

                if self.model_name == "depthanything3" and self.multiview_num_neighbors > 0:
                    batch_predictions = self._predict_da3_multiview(batch_samples, samples)
                else:
                    batch_images = [s["image_path"] for s in batch_samples]
                    batch_predictions = self._predict_batch(batch_images)

                for batch_index, sample in enumerate(batch_samples):
                    batch_pred = batch_predictions
                    if isinstance(batch_pred, dict):
                        pred_depth = batch_pred["depth"][batch_index] if batch_pred["depth"].ndim == 3 else batch_pred["depth"]
                        pred_conf = batch_pred.get("conf")
                        if pred_conf is not None:
                            pred_conf = pred_conf[batch_index] if pred_conf.ndim == 3 else pred_conf
                        else:
                            pred_conf = None
                    else:
                        pred_depth = batch_pred[batch_index]
                        pred_conf = None
                    gt_depth = _load_exr_single_channel(sample["depth_path"])
                    mask = np.ones(gt_depth.shape, dtype=np.float32)
                    pred_depth = self._prepare_prediction(pred_depth, gt_depth)

                    if pred_conf is not None and pred_conf.shape != gt_depth.shape:
                        pred_conf = cv2.resize(pred_conf, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)

                    if self.depth_align_method == "median":
                        pred_depth_aligned, depth_scale = align_depth_median(pred_depth, gt_depth)
                    elif self.depth_align_method == "least_squares":
                        pred_depth_aligned, depth_scale = align_depth_least_squares(pred_depth, gt_depth)
                    elif self.depth_align_method == "affine":
                        pred_depth_aligned, depth_scale, depth_shift = align_depth_affine(pred_depth, gt_depth)
                    elif self.depth_align_method == "multiview_metric":
                        pred_depth_aligned = pred_depth
                        depth_scale = 1.0
                    else:
                        pred_depth_aligned = pred_depth
                        depth_scale = 1.0

                    cam_accumulator.update(pred_depth_aligned, gt_depth, mask)

                    if self.save_adamvs_format and self.adamvs_output_path is not None:
                        self._save_adamvs_format(
                            pred_depth_aligned, pred_conf, gt_depth, cam_id, sample
                        )

                    if self.eval_dsm and self._name_to_params is not None and self._dsm_grid is not None:
                        image_name = "{}/{}.png".format(cam_id, sample["stem"])
                        param_pair = self._name_to_params.get(image_name)
                        if param_pair is not None:
                            cam_param, img_param = param_pair
                            downsample_factor = cam_param.width / gt_depth.shape[1]

                            elevation_error, valid_mask = compute_elevation_error_per_pixel(
                                pred_depth_aligned, gt_depth, cam_param, img_param,
                                downsample_factor=downsample_factor,
                            )
                            cam_dsm_accumulator.update(elevation_error, valid_mask)

                            pred_dsm_i = depth_to_dsm(
                                pred_depth_aligned, cam_param, img_param,
                                self._dsm_grid, downsample_factor=downsample_factor,
                            )
                            gt_dsm_i = depth_to_dsm(
                                gt_depth, cam_param, img_param,
                                self._dsm_grid, downsample_factor=downsample_factor,
                            )

                            valid_pred = np.isfinite(pred_dsm_i)
                            valid_gt = np.isfinite(gt_dsm_i)

                            if cam_pred_dsm_sum is None:
                                cam_pred_dsm_sum = np.zeros_like(pred_dsm_i, dtype=np.float64)
                                cam_pred_dsm_count = np.zeros_like(pred_dsm_i, dtype=np.float64)
                                cam_gt_dsm_sum = np.zeros_like(gt_dsm_i, dtype=np.float64)
                                cam_gt_dsm_count = np.zeros_like(gt_dsm_i, dtype=np.float64)

                            np.add.at(cam_pred_dsm_sum, np.where(valid_pred), pred_dsm_i[valid_pred])
                            np.add.at(cam_pred_dsm_count, np.where(valid_pred), 1.0)
                            np.add.at(cam_gt_dsm_sum, np.where(valid_gt), gt_dsm_i[valid_gt])
                            np.add.at(cam_gt_dsm_count, np.where(valid_gt), 1.0)
                        else:
                            logger.warning("[WARN] No camera params for image %s, skipping DSM eval", image_name)

                    if self.eval_normal and sample["normal_path"] is not None:
                        gt_normal = _load_exr_normals(sample["normal_path"])
                        gt_normal_resized = gt_normal
                        pred_normal = self._depth_to_normal(pred_depth_aligned)
                        if pred_normal.shape[:2] != gt_normal_resized.shape[:2]:
                            pred_normal = cv2.resize(
                                pred_normal, (gt_normal_resized.shape[1], gt_normal_resized.shape[0]),
                                interpolation=cv2.INTER_LINEAR,
                            )
                        normal_mask = mask
                        nm = _compute_normal_metrics(pred_normal, gt_normal_resized, normal_mask)
                        if not np.isnan(nm["mean_ang_error"]):
                            cam_normal_errors["mean_ang_error"].append(nm["mean_ang_error"])
                            cam_normal_errors["median_ang_error"].append(nm["median_ang_error"])
                            cam_normal_errors["normal_valid_count"] += nm["normal_valid_count"]

            cam_depth_result = cam_accumulator.finalize()
            cam_result = {"depth": cam_depth_result}

            if self.eval_dsm and cam_dsm_accumulator is not None:
                cam_dsm_result = cam_dsm_accumulator.finalize()
                cam_result["dsm_pixel"] = cam_dsm_result
                overall_dsm_accumulator.merge(cam_dsm_accumulator)

            if cam_pred_dsm_sum is not None and self._dsm_grid is not None:
                cam_fused_pred_dsm = np.full_like(cam_pred_dsm_sum, np.nan, dtype=np.float64)
                cam_fused_gt_dsm = np.full_like(cam_gt_dsm_sum, np.nan, dtype=np.float64)
                valid_pred = cam_pred_dsm_count > 0
                valid_gt = cam_gt_dsm_count > 0
                cam_fused_pred_dsm[valid_pred] = cam_pred_dsm_sum[valid_pred] / cam_pred_dsm_count[valid_pred]
                cam_fused_gt_dsm[valid_gt] = cam_gt_dsm_sum[valid_gt] / cam_gt_dsm_count[valid_gt]

                cam_dsm_grid_result = compute_dsm_metrics_against_gt(
                    cam_fused_pred_dsm, cam_fused_gt_dsm,
                    outlier_threshold=self.outlier_threshold,
                )
                cam_result["dsm_grid"] = cam_dsm_grid_result
                logger.info(
                    "[INFO] Camera %s DSM grid metrics: MAE=%.4f RMSE=%.4f PAG_0.2=%.2f%% PAG_0.4=%.2f%% PAG_0.6=%.2f%%",
                    cam_id, cam_dsm_grid_result["MAE"], cam_dsm_grid_result["RMSE"],
                    cam_dsm_grid_result["PAG_0.2m"], cam_dsm_grid_result["PAG_0.4m"],
                    cam_dsm_grid_result["PAG_0.6m"],
                )

                if pred_dsm_sum is None:
                    pred_dsm_sum = cam_pred_dsm_sum.copy()
                    pred_dsm_count = cam_pred_dsm_count.copy()
                    gt_dsm_sum = cam_gt_dsm_sum.copy()
                    gt_dsm_count = cam_gt_dsm_count.copy()
                else:
                    np.add.at(pred_dsm_sum, np.where(valid_pred), cam_pred_dsm_sum[valid_pred])
                    np.add.at(pred_dsm_count, np.where(valid_pred), cam_pred_dsm_count[valid_pred])
                    np.add.at(gt_dsm_sum, np.where(valid_gt), cam_gt_dsm_sum[valid_gt])
                    np.add.at(gt_dsm_count, np.where(valid_gt), cam_gt_dsm_count[valid_gt])

            if self.eval_normal and cam_normal_errors["mean_ang_error"]:
                cam_result["normal"] = {
                    "mean_ang_error": float(np.mean(cam_normal_errors["mean_ang_error"])),
                    "median_ang_error": float(np.median(cam_normal_errors["median_ang_error"])),
                    "normal_valid_count": cam_normal_errors["normal_valid_count"],
                }
                overall_normal_metrics["mean_ang_error"].extend(cam_normal_errors["mean_ang_error"])
                overall_normal_metrics["median_ang_error"].extend(cam_normal_errors["median_ang_error"])
                overall_normal_metrics["normal_valid_count"] += cam_normal_errors["normal_valid_count"]

            per_camera_results["cam{}".format(cam_id)] = cam_result
            overall_accumulator.merge(cam_accumulator)

        overall_depth_result = overall_accumulator.finalize()
        overall_result = {"depth": overall_depth_result}

        if self.eval_dsm and overall_dsm_accumulator is not None:
            overall_dsm_result = overall_dsm_accumulator.finalize()
            overall_result["dsm_pixel"] = overall_dsm_result

        if pred_dsm_sum is not None and self._dsm_grid is not None:
            fused_pred_dsm = np.full_like(pred_dsm_sum, np.nan, dtype=np.float64)
            fused_gt_dsm = np.full_like(gt_dsm_sum, np.nan, dtype=np.float64)
            valid_pred = pred_dsm_count > 0
            valid_gt = gt_dsm_count > 0
            fused_pred_dsm[valid_pred] = pred_dsm_sum[valid_pred] / pred_dsm_count[valid_pred]
            fused_gt_dsm[valid_gt] = gt_dsm_sum[valid_gt] / gt_dsm_count[valid_gt]

            overall_dsm_grid_result = compute_dsm_metrics_against_gt(
                fused_pred_dsm, fused_gt_dsm,
                outlier_threshold=self.outlier_threshold,
            )
            overall_result["dsm_grid"] = overall_dsm_grid_result

            gt_dsm_data = self._dsm_grid.data.astype(np.float64)
            gt_dsm_grid_result = compute_dsm_metrics_against_gt(
                fused_pred_dsm, gt_dsm_data,
                outlier_threshold=self.outlier_threshold,
            )
            overall_result["dsm_grid_vs_gt_file"] = gt_dsm_grid_result

            logger.info(
                "[INFO] Overall DSM grid metrics (fused pred vs fused GT): MAE=%.4f RMSE=%.4f PAG_0.2=%.2f%% PAG_0.4=%.2f%% PAG_0.6=%.2f%%",
                overall_dsm_grid_result["MAE"], overall_dsm_grid_result["RMSE"],
                overall_dsm_grid_result["PAG_0.2m"], overall_dsm_grid_result["PAG_0.4m"],
                overall_dsm_grid_result["PAG_0.6m"],
            )
            logger.info(
                "[INFO] Overall DSM grid metrics (fused pred vs GT DSM file): MAE=%.4f RMSE=%.4f PAG_0.2=%.2f%% PAG_0.4=%.2f%% PAG_0.6=%.2f%%",
                gt_dsm_grid_result["MAE"], gt_dsm_grid_result["RMSE"],
                gt_dsm_grid_result["PAG_0.2m"], gt_dsm_grid_result["PAG_0.4m"],
                gt_dsm_grid_result["PAG_0.6m"],
            )

        if self.eval_normal and overall_normal_metrics["mean_ang_error"]:
            overall_result["normal"] = {
                "mean_ang_error": float(np.mean(overall_normal_metrics["mean_ang_error"])),
                "median_ang_error": float(np.median(overall_normal_metrics["median_ang_error"])),
                "normal_valid_count": overall_normal_metrics["normal_valid_count"],
            }

        return {
            "model_name": self.model_name,
            "dataset_path": str(self.dataset_path),
            "camera_ids": self.camera_ids,
            "align_mode": self.align_mode,
            "outlier_threshold": self.outlier_threshold,
            "per_camera": per_camera_results,
            "overall": overall_result,
        }

    def _save_adamvs_format(
        self,
        pred_depth: np.ndarray,
        pred_conf: Optional[np.ndarray],
        gt_depth: np.ndarray,
        cam_id: str,
        sample: Dict,
    ) -> None:
        stem = sample["stem"]
        image_path = sample["image_path"]
        cam_dir = self.adamvs_output_path / cam_id
        cam_dir.mkdir(parents=True, exist_ok=True)

        save_pfm(str(cam_dir / "{}_init.pfm".format(stem)), pred_depth.astype(np.float32))

        if pred_conf is not None:
            conf_save = pred_conf.astype(np.float32)
            if conf_save.max() > 1.0:
                conf_save = conf_save / conf_save.max()
            save_pfm(str(cam_dir / "{}_prob.pfm".format(stem)), conf_save)
        else:
            ones_conf = np.ones_like(pred_depth, dtype=np.float32)
            save_pfm(str(cam_dir / "{}_prob.pfm".format(stem)), ones_conf)

        image_name = "{}/{}.png".format(cam_id, stem)
        param_pair = self._name_to_params.get(image_name) if self._name_to_params else None

        if param_pair is not None:
            cam_param, img_param = param_pair
            O_xrightyup = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
            Rwc_down = img_param.Rwc @ O_xrightyup
            extrinsic_wc = np.eye(4, dtype=np.float64)
            extrinsic_wc[:3, :3] = Rwc_down
            extrinsic_wc[:3, 3] = img_param.twc
            extrinsic_cw = np.linalg.inv(extrinsic_wc)

            intrinsic = np.zeros((3, 3), dtype=np.float64)
            intrinsic[0, 0] = cam_param.fx
            intrinsic[1, 1] = cam_param.fy
            intrinsic[0, 2] = cam_param.cx
            intrinsic[1, 2] = cam_param.cy
            intrinsic[2, 2] = 1.0

            depth_min = float(img_param.min_depth) if img_param.min_depth > 0 else float(np.nanmin(pred_depth[pred_depth > 1e-8]))
            depth_max = float(img_param.max_depth) if img_param.max_depth > 0 else float(np.nanmax(pred_depth[pred_depth > 1e-8]))
        else:
            extrinsic_cw = np.eye(4, dtype=np.float64)
            intrinsic = np.eye(3, dtype=np.float64)
            valid_depth = pred_depth[pred_depth > 1e-8]
            depth_min = float(np.nanmin(valid_depth)) if len(valid_depth) > 0 else 0.0
            depth_max = float(np.nanmax(valid_depth)) if len(valid_depth) > 0 else 1.0

        write_adamvs_cam(
            str(cam_dir / "{}.txt".format(stem)),
            extrinsic_cw.astype(np.float32),
            intrinsic.astype(np.float32),
            depth_min,
            depth_max,
            num_depth=192,
            ref_image_path=str(image_path),
        )

        try:
            img = Image.open(str(image_path))
            img.save(str(cam_dir / "{}.jpg".format(stem)), "JPEG", quality=95)
        except Exception as e:
            logger.warning("[WARN] Failed to save reference image %s: %s", image_path, e)

    @staticmethod
    def _depth_to_normal(depth_map: np.ndarray) -> np.ndarray:
        dzdx = np.zeros_like(depth_map)
        dzdy = np.zeros_like(depth_map)
        dzdx[:, 1:-1] = (depth_map[:, 2:] - depth_map[:, :-2]) / 2.0
        dzdy[1:-1, :] = (depth_map[2:, :] - depth_map[:-2, :]) / 2.0

        normal = np.zeros((*depth_map.shape, 3), dtype=np.float32)
        normal[:, :, 0] = -dzdx
        normal[:, :, 1] = -dzdy
        normal[:, :, 2] = 1.0

        norm = np.linalg.norm(normal, axis=-1, keepdims=True).clip(min=1e-8)
        normal = normal / norm
        return normal


def _format_metric(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    return "{:.6f}".format(value)


def _print_results(results):
    overall = results["overall"]
    depth = overall["depth"]

    ada_mvs_headers = ["MAE", "RMSE", "PAG_0.2m", "PAG_0.4m", "PAG_0.6m"]
    standard_headers = ["abs_rel", "sq_rel", "rmse_log", "log10", "silog", "delta1", "delta2", "delta3"]

    print("=" * 120)
    print("WHU-OMVS Predict Split Benchmark Metrics (Ada-MVS / Liu et al. 2023)")
    print("=" * 120)
    print(
        "model={} cameras={} align={} outlier_T={}m".format(
            results["model_name"], results["camera_ids"], results["align_mode"],
            results.get("outlier_threshold", 20.0),
        )
    )
    print()

    if "dsm_grid_vs_gt_file" in overall:
        dsm = overall["dsm_grid_vs_gt_file"]
        print("=== DSM Grid Metrics vs GT DSM File (Paper Table 1 benchmark) ===")
        print("  Fused predicted DSM compared against the official GT DSM raster file.")
        print("  This is the CLOSEST match to the Ada-MVS paper evaluation pipeline:")
        print("    depth maps -> 3D points -> fuse -> DSM grid -> compare with GT DSM")
        print("  PAG thresholds (0.2m/0.4m/0.6m) correspond to 1x/2x/3x GSD (0.2m).")
        print("  MAE/RMSE: outlier threshold T = 20m (100x GSD), excluded from denominator.")
        print("  PAG: denominator = all valid GT grid cells (outliers NOT excluded).")
        print()
        print("  " + " | ".join("{:>12}".format(h) for h in ada_mvs_headers))
        print("  " + " | ".join("{:>12}".format(_format_metric(dsm[h])) for h in ada_mvs_headers))
        print("  valid_grid_cells={} outlier_cells={}".format(dsm["valid_count"], dsm.get("outlier_count", "N/A")))
        print()

    if "dsm_grid" in overall:
        dsm = overall["dsm_grid"]
        print("=== DSM Grid Metrics vs Fused GT DSM (from depth unprojection) ===")
        print("  Fused predicted DSM compared against fused GT DSM (from depth map unprojection).")
        print("  Both pred and GT DSMs are generated by unprojecting depth maps to 3D and")
        print("  rasterizing onto the same grid. Serves as a cross-check.")
        print()
        print("  " + " | ".join("{:>12}".format(h) for h in ada_mvs_headers))
        print("  " + " | ".join("{:>12}".format(_format_metric(dsm[h])) for h in ada_mvs_headers))
        print("  valid_grid_cells={} outlier_cells={}".format(dsm["valid_count"], dsm.get("outlier_count", "N/A")))
        print()

    if "dsm_pixel" in overall:
        dsm = overall["dsm_pixel"]
        print("--- DSM Pixel-Level Metrics (per-pixel elevation error, accumulated) ---")
        print("  Per-pixel elevation error (pred_z - gt_z) accumulated across all images.")
        print("  This is NOT the same as the paper's DSM grid comparison, but provides")
        print("  a per-pixel view of the elevation error distribution.")
        print()
        print("  " + " | ".join("{:>12}".format(h) for h in ada_mvs_headers))
        print("  " + " | ".join("{:>12}".format(_format_metric(dsm[h])) for h in ada_mvs_headers))
        print("  valid_pixels={} outlier_pixels={}".format(dsm["valid_count"], dsm.get("outlier_count", "N/A")))
        print()

    print("--- Depth-Level Metrics (Raw Depth Map Comparison, for reference only) ---")
    print("  WARNING: These metrics compare raw depth values (528-914m range).")
    print("  PAG values here are NOT meaningful for the Ada-MVS paper benchmark.")
    print("  Use DSM Grid metrics above for paper comparison.")
    print()
    print("  " + " | ".join("{:>12}".format(h) for h in ada_mvs_headers))
    print("  " + " | ".join("{:>12}".format(_format_metric(depth[h])) for h in ada_mvs_headers))
    print("  valid_pixels={} outlier_pixels={}".format(depth["valid_count"], depth.get("outlier_count", "N/A")))
    print()

    print("--- Standard Depth Metrics (for cross-benchmark comparison) ---")
    print("  " + " | ".join("{:>12}".format(h) for h in standard_headers))
    print("  " + " | ".join("{:>12}".format(_format_metric(depth[h])) for h in standard_headers))
    print("-" * 120)

    if "normal" in overall:
        nm = overall["normal"]
        print("Normal: mean_ang={:.4f} deg, median_ang={:.4f} deg, valid={}".format(
            nm["mean_ang_error"], nm["median_ang_error"], nm["normal_valid_count"]
        ))
        print("-" * 120)

    for cam_key, cam_result in results["per_camera"].items():
        cam_depth = cam_result["depth"]
        print()
        print("  {}:".format(cam_key))
        if "dsm_grid" in cam_result:
            cam_dsm = cam_result["dsm_grid"]
            print("    DSM Grid:  " + " | ".join("{:>12}".format(_format_metric(cam_dsm[h])) for h in ada_mvs_headers))
        if "dsm_pixel" in cam_result:
            cam_dsm = cam_result["dsm_pixel"]
            print("    DSM Pixel: " + " | ".join("{:>12}".format(_format_metric(cam_dsm[h])) for h in ada_mvs_headers))
        print("    Depth:     " + " | ".join("{:>12}".format(_format_metric(cam_depth[h])) for h in ada_mvs_headers))
        print("    Standard:  " + " | ".join("{:>12}".format(_format_metric(cam_depth[h])) for h in standard_headers))
        if "normal" in cam_result:
            cn = cam_result["normal"]
            print("    Normal:    mean_ang={:.4f} deg, median_ang={:.4f} deg".format(
                cn["mean_ang_error"], cn["median_ang_error"]
            ))


def parse_args():
    parser = argparse.ArgumentParser(
        description="WHU-OMVS predict split metric inference (Ada-MVS benchmark metrics)"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(_REPO_ROOT / "configs" / "base_config.yaml"),
        help="Path to the model config file.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(_REPO_ROOT / "dataset" / "WHU-OMVS" / "predict"),
        help="Path to the WHU-OMVS predict directory.",
    )
    parser.add_argument(
        "--camera_ids",
        nargs="*",
        default=None,
        help="Camera IDs to evaluate. Defaults to [1,2,3,4,5].",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Number of images per inference batch.")
    parser.add_argument(
        "--align_mode",
        type=str,
        default="median",
        choices=["none", "median"],
        help="Per-image scale alignment before metric computation.",
    )
    parser.add_argument(
        "--outlier_threshold",
        type=float,
        default=20.0,
        help="Outlier threshold T in meters for MAE/RMSE (default: 20m = 100*GSD, GSD=0.2m).",
    )
    parser.add_argument(
        "--eval_normal",
        action="store_true",
        help="Also evaluate normal metrics (depth-derived vs GT normals).",
    )
    parser.add_argument(
        "--eval_dsm",
        action="store_true",
        default=True,
        help="Evaluate DSM-level metrics (elevation error in world coordinates). Default: True.",
    )
    parser.add_argument(
        "--no_eval_dsm",
        action="store_true",
        help="Disable DSM-level evaluation.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(_REPO_ROOT / "exp" / "whu-omvs" / "predict_metric_eval"),
        help="Directory for saving the metrics json.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        choices=["depthanything3", "mapanything", "pi3", "vggt"],
        help="Model name override. If provided, it supersedes Model.name in config.",
    )
    parser.add_argument(
        "--save_adamvs_format",
        action="store_true",
        help="Save predictions in Ada-MVS output format (PFM depth/conf, TXT cam params, JPG image).",
    )
    parser.add_argument(
        "--adamvs_output_path",
        type=str,
        default=None,
        help="Output directory for Ada-MVS format files. Defaults to <output_path>/adamvs_output.",
    )
    parser.add_argument(
        "--depth_align_method",
        type=str,
        default="median",
        choices=["none", "median", "least_squares", "affine", "multiview_metric"],
        help="Method to align relative depth to absolute metric depth using GT. 'affine' uses GT=a*pred+b. 'multiview_metric' uses DA3 multi-view metric depth (no GT alignment).",
    )
    parser.add_argument(
        "--multiview_num_neighbors",
        type=int,
        default=0,
        help="Number of neighboring views for DA3 multi-view inference. 0=disabled (single-view). E.g., 5 means 2 before + target + 2 after.",
    )
    parser.add_argument(
        "--multiview_process_res",
        type=int,
        default=504,
        help="Processing resolution for DA3 multi-view inference. Default: 504.",
    )
    parser.add_argument(
        "--multiview_ref_strategy",
        type=str,
        default="saddle_balanced",
        choices=["first", "middle", "saddle_balanced", "saddle_sim_range"],
        help="Reference view strategy for DA3 multi-view inference. Default: saddle_balanced.",
    )
    parser.add_argument("--log_level", type=str, default="INFO", help="Python logging level.")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    config_path = _resolve_path(args.config_path)
    dataset_path = _resolve_path(args.dataset_path)
    output_path = Path(_resolve_path(args.output_path))
    output_path.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)
    if args.model_name is not None:
        if "Model" not in cfg or not isinstance(cfg["Model"], dict):
            cfg["Model"] = {}
        cfg["Model"]["name"] = args.model_name

    camera_ids = args.camera_ids if args.camera_ids else ["1", "2", "3", "4", "5"]
    eval_dsm = args.eval_dsm and not args.no_eval_dsm

    adamvs_output_path = args.adamvs_output_path
    if adamvs_output_path is None and args.save_adamvs_format:
        adamvs_output_path = str(output_path / "adamvs_output")

    runner = PredictMetricInfer(
        cfg=cfg,
        dataset_path=dataset_path,
        camera_ids=camera_ids,
        batch_size=args.batch_size,
        align_mode=args.align_mode,
        eval_normal=args.eval_normal,
        outlier_threshold=args.outlier_threshold,
        eval_dsm=eval_dsm,
        save_adamvs_format=args.save_adamvs_format,
        adamvs_output_path=adamvs_output_path,
        depth_align_method=args.depth_align_method,
        multiview_num_neighbors=args.multiview_num_neighbors,
        multiview_process_res=args.multiview_process_res,
        multiview_ref_strategy=args.multiview_ref_strategy,
    )
    results = runner.run()
    _print_results(results)

    cam_suffix = "_".join("cam{}".format(c) for c in camera_ids)
    output_file = output_path / "predict_{}_{}_metrics.json".format(cam_suffix, results["model_name"])
    with open(output_file, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("[INFO] Metrics saved to %s", output_file)


if __name__ == "__main__":
    main()
