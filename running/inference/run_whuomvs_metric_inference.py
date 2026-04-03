import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

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
for path in (_REPO_ROOT, _RUNNING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from metrics import DepthMetricAccumulator
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
    parts = path.stem.split("_")
    if len(parts) >= 2 and all(part.isdigit() for part in parts[:2]):
        return int(parts[0]), int(parts[1])
    if path.stem.isdigit():
        return int(path.stem), 0
    return path.stem


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


def _load_mask(path: Optional[Path], shape) -> np.ndarray:
    if path is None or not path.exists():
        return np.ones(shape, dtype=np.float32)
    mask = Image.open(path).convert("L")
    mask = np.asarray(mask, dtype=np.float32) / 255.0
    return (mask > 0.5).astype(np.float32)


class MetricInfer:
    """Run DA3 inference on WHU-OMVS test split and compute depth metrics."""

    def __init__(self, cfg, dataset_path, split, areas, camera_id, batch_size, align_mode):
        self.cfg = cfg
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.areas = areas
        self.camera_id = str(camera_id)
        self.batch_size = max(1, int(batch_size))
        self.align_mode = align_mode
        self.model_name = self.cfg["Model"].get("name", self.cfg["Model"].get("model_name", "depthanything3"))
        if torch.cuda.is_available():
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

    def _cfg_get_weight(self, model_key, field, fallback_flat_key=None):
        weights = self.cfg.get("Weights", {})
        if model_key in weights and isinstance(weights[model_key], dict) and field in weights[model_key]:
            return weights[model_key][field]
        if fallback_flat_key is not None and fallback_flat_key in weights:
            return weights[fallback_flat_key]
        raise KeyError(f"Missing weight config for {model_key}.{field} (fallback={fallback_flat_key})")

    def _load_model(self):
        if self.model_name == "depthanything3":
            da3_config_path = self._cfg_get_weight("depthanything3", "DA3_CONFIG", fallback_flat_key="DA3_CONFIG")
            da3_weight_path = self._cfg_get_weight("depthanything3", "DA3", fallback_flat_key="DA3")
            with open(da3_config_path, "r", encoding="utf-8") as file_handle:
                da3_config = json.load(file_handle)
            model = DepthAnything3(**da3_config)
            state_dict = load_file(da3_weight_path)
            model.load_state_dict(state_dict, strict=False)
        elif self.model_name == "mapanything":
            map_config_path = self._cfg_get_weight("mapanything", "MAP_CONFIG", fallback_flat_key="MAP_CONFIG")
            map_weight_path = self._cfg_get_weight("mapanything", "MAP", fallback_flat_key="MAP")
            with open(map_config_path, "r", encoding="utf-8") as file_handle:
                map_config = json.load(file_handle)
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

    def _resolve_areas(self):
        if self.areas:
            return self.areas
        index_file = self.dataset_path / self.split / "index.txt"
        with open(index_file, "r", encoding="utf-8") as file_handle:
            return [line.strip() for line in file_handle.readlines() if line.strip()]

    def _collect_samples_for_area(self, area_name):
        area_dir = self.dataset_path / self.split / area_name
        image_dir = area_dir / "images" / self.camera_id
        depth_dir = area_dir / "depths" / self.camera_id
        mask_dir = area_dir / "masks" / self.camera_id

        if not image_dir.is_dir() or not depth_dir.is_dir():
            logger.warning("[WARN] Skip missing area: %s", area_dir)
            return []

        image_files = {path.stem: path for path in sorted(image_dir.glob("*.png"), key=_parse_stem_key)}
        depth_files = {path.stem: path for path in sorted(depth_dir.glob("*.exr"), key=_parse_stem_key)}
        mask_files = {path.stem: path for path in sorted(mask_dir.glob("*.png"), key=_parse_stem_key)} if mask_dir.is_dir() else {}

        common_stems = sorted(set(image_files) & set(depth_files), key=lambda value: tuple(int(part) for part in value.split("_")))
        samples = []
        for stem in common_stems:
            samples.append(
                {
                    "stem": stem,
                    "image_path": image_files[stem],
                    "depth_path": depth_files[stem],
                    "mask_path": mask_files.get(stem),
                }
            )
        return samples

    def _predict_batch(self, image_paths):
        if self.model_name == "mapanything":
            return self._predict_batch_mapanything(image_paths)
        if self.model_name == "pi3":
            return self._predict_batch_pi3(image_paths)
        if self.model_name == "vggt":
            return self._predict_batch_vggt(image_paths)

        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.amp.autocast("cuda", dtype=self.dtype):
                    prediction = self.model.inference([str(path) for path in image_paths])
            else:
                prediction = self.model.inference([str(path) for path in image_paths])

        depth = prediction.depth
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
        else:
            depth = np.asarray(depth)
        if depth.ndim == 4 and depth.shape[1] == 1:
            depth = np.squeeze(depth, axis=1)
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = np.squeeze(depth, axis=-1)
        return depth

    def _predict_batch_mapanything(self, image_paths):
        outputs = []
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
            depth = np.asarray(depth)
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = np.squeeze(depth, axis=-1)
            outputs.append(depth)
        return np.stack(outputs, axis=0)

    def _predict_batch_pi3(self, image_paths):
        outputs = []
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
            outputs.append(depth)
        return np.stack(outputs, axis=0)

    def _predict_batch_vggt(self, image_paths):
        outputs = []
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
            outputs.append(depth)
        return np.stack(outputs, axis=0)

    def _prepare_prediction(self, pred_depth, gt_depth):
        pred_depth = np.asarray(pred_depth, dtype=np.float32)
        gt_depth = np.asarray(gt_depth, dtype=np.float32)
        if pred_depth.shape != gt_depth.shape:
            pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        return pred_depth

    def run(self):
        per_area_results = {}
        overall_accumulator = DepthMetricAccumulator(align_mode=self.align_mode)

        for area_name in self._resolve_areas():
            samples = self._collect_samples_for_area(area_name)
            if not samples:
                logger.warning("[WARN] No samples found for area %s", area_name)
                continue

            area_accumulator = DepthMetricAccumulator(align_mode=self.align_mode)
            logger.info(
                "[INFO] Evaluating area=%s camera=%s samples=%d",
                area_name,
                self.camera_id,
                len(samples),
            )

            for start_idx in range(0, len(samples), self.batch_size):
                batch_samples = samples[start_idx : start_idx + self.batch_size]
                batch_images = [sample["image_path"] for sample in batch_samples]
                batch_predictions = self._predict_batch(batch_images)

                for batch_index, sample in enumerate(batch_samples):
                    pred_depth = batch_predictions[batch_index]
                    gt_depth = _load_exr_single_channel(sample["depth_path"])
                    mask = _load_mask(sample["mask_path"], gt_depth.shape)
                    pred_depth = self._prepare_prediction(pred_depth, gt_depth)
                    area_accumulator.update(pred_depth, gt_depth, mask)

            area_result = area_accumulator.finalize()
            per_area_results[area_name] = area_result
            overall_accumulator.merge(area_accumulator)

        overall_result = overall_accumulator.finalize()
        return {
            "model_name": self.model_name,
            "split": self.split,
            "dataset_path": str(self.dataset_path),
            "camera_id": self.camera_id,
            "align_mode": self.align_mode,
            "areas": per_area_results,
            "overall": overall_result,
        }


def _format_metric(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    return f"{value:.6f}"


def _print_results(results):
    overall = results["overall"]
    print("=" * 92)
    print("WHU-OMVS Depth Metrics")
    print("=" * 92)
    print(
        f"model={results['model_name']} split={results['split']} camera={results['camera_id']} align={results['align_mode']}"
    )
    print(f"valid_pixels={overall['valid_count']}")
    print("-" * 92)
    headers = ["abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog", "delta1", "delta2", "delta3"]
    print(" | ".join(f"{header:>10}" for header in headers))
    print(" | ".join(f"{_format_metric(overall[header]):>10}" for header in headers))
    print("-" * 92)
    for area_name, area_result in results["areas"].items():
        values = " | ".join(f"{_format_metric(area_result[header]):>10}" for header in headers)
        print(f"{area_name:>10}: {values}")


def parse_args():
    parser = argparse.ArgumentParser(description="WHU-OMVS metric inference")
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(_REPO_ROOT / "configs" / "base_config.yaml"),
        help="Path to the model config file.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(_REPO_ROOT / "dataset" / "WHU-OMVS"),
        help="Path to the WHU-OMVS dataset root.",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate.")
    parser.add_argument(
        "--areas",
        nargs="*",
        default=None,
        help="Optional list of area names. Defaults to the split index file.",
    )
    parser.add_argument("--camera_id", type=int, default=3, help="Single camera id to evaluate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of images per inference batch.")
    parser.add_argument(
        "--align_mode",
        type=str,
        default="none",
        choices=["none", "median"],
        help="Optional per-image scale alignment before metric computation.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(_REPO_ROOT / "exp" / "whu-omvs" / "metric_eval"),
        help="Directory for saving the metrics json.",
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
    runner = MetricInfer(
        cfg=cfg,
        dataset_path=dataset_path,
        split=args.split,
        areas=args.areas,
        camera_id=args.camera_id,
        batch_size=args.batch_size,
        align_mode=args.align_mode,
    )
    results = runner.run()
    _print_results(results)

    output_file = output_path / f"{results['split']}_cam{results['camera_id']}_{results['model_name']}_metrics.json"
    with open(output_file, "w", encoding="utf-8") as file_handle:
        json.dump(results, file_handle, indent=2)
    logger.info("[INFO] Metrics saved to %s", output_file)


if __name__ == "__main__":
    main()

