from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import cv2
import Imath
import OpenEXR
import numpy as np
import torch
from PIL import Image


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "models").is_dir() and (p / "running").is_dir() and (p / "configs").is_dir():
            return p
    return start.parent


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _find_repo_root(_SCRIPT_DIR)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from running.training.run_train_da3mvs_whuomvs import (
    AdaMVSFeatureEncoder,
    DA3MVSForMetricDepth,
    DA3MVSFusionHead,
    LayerAttentionFusionHead,
    MetricScaleShift,
    build_da3_model,
)
from running.metrics.accumulator import DepthMetricAccumulator
from running.utils.viz_utils import depth_to_color

logger = logging.getLogger(__name__)


def _resolve_to_repo_path(path_value: str | None) -> str | None:
    if path_value is None:
        return None
    p = Path(path_value)
    if p.is_absolute():
        return str(p)
    return str((_REPO_ROOT / p).resolve())


def _load_rgb_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def _resize_longest_side(image: Image.Image, target_size: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest == target_size:
        return image
    scale = target_size / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = Image.Resampling.BICUBIC if scale > 1.0 else Image.Resampling.BILINEAR
    return image.resize((new_width, new_height), interpolation)


def _resize_shortest_side(image: Image.Image, target_size: int) -> Image.Image:
    width, height = image.size
    shortest = min(width, height)
    if shortest == target_size:
        return image
    scale = target_size / float(shortest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = Image.Resampling.BICUBIC if scale > 1.0 else Image.Resampling.BILINEAR
    return image.resize((new_width, new_height), interpolation)


def _resize_square(image: Image.Image, target_size: int) -> Image.Image:
    width, height = image.size
    max_dim = max(width, height)
    left = (max_dim - width) // 2
    top = (max_dim - height) // 2
    square = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    square.paste(image, (left, top))
    return square.resize((target_size, target_size), Image.Resampling.BICUBIC)


def _preprocess_image(image_path: str, target_size: int, process_res_method: str) -> tuple[torch.Tensor, np.ndarray]:
    image = _load_rgb_image(image_path)
    if process_res_method == "square":
        image = _resize_square(image, target_size)
    elif process_res_method in ("upper_bound_resize", "upper_bound_crop"):
        image = _resize_longest_side(image, target_size)
    elif process_res_method in ("lower_bound_resize", "lower_bound_crop"):
        image = _resize_shortest_side(image, target_size)
    else:
        raise ValueError(f"Unsupported process_res_method: {process_res_method}")

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
    return image_tensor, image_array


def _as_spatial_map(value: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    return value


def _load_exr_depth(exr_path: Path) -> np.ndarray:
    exr_file = OpenEXR.InputFile(str(exr_path))
    header = exr_file.header()
    data_window = header["dataWindow"]
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_bytes = exr_file.channel("Y", pixel_type)
    return np.frombuffer(depth_bytes, dtype=np.float32).reshape(height, width).copy()


def _load_mask(mask_path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if mask_path is None or not mask_path.exists():
        return np.ones(shape, dtype=np.float32)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.ones(shape, dtype=np.float32)
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.float32)


def _resolve_gt_paths(split: str, dataset_root: Path, area_name: str | None, camera_id: str, stem: str) -> tuple[Path | None, Path | None]:
    if split == "test":
        if area_name is None:
            return None, None
        gt_depth_path = dataset_root / "test" / area_name / "depths" / camera_id / f"{stem}.exr"
        mask_path = dataset_root / "test" / area_name / "masks" / camera_id / f"{stem}.png"
        return gt_depth_path, mask_path
    gt_depth_path = dataset_root / "predict" / "GT" / "GT_Depths" / camera_id / f"{stem}.exr"
    return gt_depth_path, None


def _iter_image_paths(root_dir: Path) -> Iterable[Path]:
    for path in sorted(root_dir.glob("*.png")):
        yield path
    for path in sorted(root_dir.glob("*.jpg")):
        yield path
    for path in sorted(root_dir.glob("*.jpeg")):
        yield path


def _load_da3mvs_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = dict(ckpt_args)

    model_name = ckpt_args.get("model_name", "da3-large")
    pretrained_path = ckpt_args.get("pretrained_path", None)
    adamvs_ckpt = ckpt_args.get("adamvs_ckpt", None)
    adamvs_feature_stage = ckpt_args.get("adamvs_feature_stage", "stage3")
    fusion_dim = int(ckpt_args.get("fusion_dim", 128))
    fusion_type = ckpt_args.get("fusion_type", "layer_attention")
    train_adamvs = bool(ckpt_args.get("train_adamvs", False))

    da3_model = build_da3_model(model_name=model_name, pretrained_path=pretrained_path)
    scale_shift = MetricScaleShift(init_scale=1.0, init_shift=0.0)
    adamvs_encoder = AdaMVSFeatureEncoder(
        ckpt_path=_resolve_to_repo_path(adamvs_ckpt),
        stage_key=adamvs_feature_stage,
        trainable=train_adamvs,
    )
    adamvs_stage_channels = {"stage1": 32, "stage2": 16, "stage3": 8}
    ada_in_dim = adamvs_stage_channels.get(adamvs_feature_stage, 8)
    if fusion_type == "layer_attention":
        fusion_head = LayerAttentionFusionHead(
            da3_in_dim=128,
            ada_in_dim=ada_in_dim,
            fusion_dim=fusion_dim,
        )
    else:
        fusion_head = DA3MVSFusionHead(
            da3_in_dim=128,
            ada_in_dim=ada_in_dim,
            fusion_dim=fusion_dim,
        )

    model = DA3MVSForMetricDepth(da3_model, scale_shift, adamvs_encoder, fusion_head)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        logger.warning("Missing keys when loading DA3MVS checkpoint (%d): %s", len(missing), missing[:5])
    if unexpected:
        logger.warning("Unexpected keys when loading DA3MVS checkpoint (%d): %s", len(unexpected), unexpected[:5])

    model = model.to(device)
    model.eval()
    return model


def _save_depth_png(depth_map: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    depth_color = depth_to_color(depth_map)
    cv2.imwrite(str(output_path), depth_color)


def _init_metric_state() -> dict:
    return {
        "overall": DepthMetricAccumulator(align_mode="median"),
        "per_area": {},
        "per_camera": {},
    }


def _update_metric_state(metric_state: dict, area_name: str | None, camera_id: str, pred_depth: np.ndarray, gt_depth: np.ndarray, mask: np.ndarray | None) -> None:
    if area_name is not None:
        area_acc = metric_state["per_area"].setdefault(area_name, DepthMetricAccumulator(align_mode="median"))
    else:
        area_acc = None
    cam_acc = metric_state["per_camera"].setdefault(camera_id, DepthMetricAccumulator(align_mode="median"))

    if pred_depth.shape != gt_depth.shape:
        pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)

    valid_mask = np.isfinite(pred_depth) & np.isfinite(gt_depth) & (pred_depth > 1e-8) & (gt_depth > 1e-8)
    if mask is not None:
        valid_mask &= mask > 0.5
    if not np.any(valid_mask):
        return

    metric_state["overall"].update(pred_depth, gt_depth, valid_mask)
    cam_acc.update(pred_depth, gt_depth, valid_mask)
    if area_acc is not None:
        area_acc.update(pred_depth, gt_depth, valid_mask)


def _save_metric_summary(output_path: Path, dataset_root: Path, split: str, areas: list[str], metric_state: dict, align_mode: str = "median") -> Path:
    per_area = {area_name: {"depth": metric_state["per_area"][area_name].finalize()} for area_name in sorted(metric_state["per_area"].keys())}
    per_camera = {f"cam{cam_id}": {"depth": metric_state["per_camera"][cam_id].finalize()} for cam_id in sorted(metric_state["per_camera"].keys(), key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value)))}
    overall_depth = metric_state["overall"].finalize()
    result = {
        "model_name": "da3mvs",
        "split": split,
        "dataset_path": str((dataset_root / split).resolve()),
        "camera_ids": sorted([cam_key.replace("cam", "") for cam_key in per_camera.keys()], key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value))),
        "camera_id": "all",
        "align_mode": align_mode,
        "per_camera": per_camera,
        "per_area": per_area,
        "overall": {"depth": overall_depth},
    }

    cam_suffix = "_".join(f"cam{cam_id}" for cam_id in result["camera_ids"])
    area_suffix = ""
    if areas:
        area_suffix = "_" + "_".join(sorted(areas))
    metrics_path = output_path / f"{split}_{cam_suffix}_da3mvs{area_suffix}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    logger.info("Saved combined depth metrics to %s", metrics_path)
    return metrics_path


def _run_one_camera(
    model,
    image_paths: list[Path],
    output_dir: Path,
    device: torch.device,
    process_res: int,
    process_res_method: str,
    batch_size: int,
    viz_max_frames: int,
    metric_state: dict | None,
    split: str,
    dataset_root: Path,
    area_name: str | None,
    camera_id: str,
) -> None:
    if len(image_paths) == 0:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        batch_tensors = []
        batch_stems = []
        for image_path in batch_paths:
            image_tensor, _ = _preprocess_image(str(image_path), process_res, process_res_method)
            batch_tensors.append(image_tensor)
            batch_stems.append(image_path.stem)

        if len(batch_tensors) == 0:
            continue

        images = torch.stack(batch_tensors, dim=0).to(device)
        images_input = images.unsqueeze(1)

        with torch.no_grad():
            autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            if torch.cuda.is_available():
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    pred_depth, pred_metric, pred_conf, _, _ = model(images_input)
            else:
                pred_depth, pred_metric, pred_conf, _, _ = model(images_input)

        depth_maps = _as_spatial_map(pred_metric if pred_metric is not None else pred_depth)
        conf_maps = _as_spatial_map(pred_conf)

        if depth_maps is None:
            logger.warning("No depth predictions returned for batch starting at %d", start)
            continue

        if depth_maps.ndim == 2:
            depth_maps = depth_maps[None, ...]
        if conf_maps is not None and conf_maps.ndim == 2:
            conf_maps = conf_maps[None, ...]

        for idx, stem in enumerate(batch_stems):
            if viz_max_frames > 0 and saved >= viz_max_frames:
                continue
            depth_map = depth_maps[idx]
            if metric_state is not None:
                gt_depth_path, mask_path = _resolve_gt_paths(split, dataset_root, area_name, camera_id, stem)
                if gt_depth_path is not None and gt_depth_path.exists():
                    gt_depth = _load_exr_depth(gt_depth_path)
                    mask = _load_mask(mask_path, gt_depth.shape) if split == "test" else None
                    _update_metric_state(metric_state, area_name, camera_id, depth_map, gt_depth, mask)
            _save_depth_png(depth_map, output_dir / f"{stem}_depth.png")
            saved += 1

    logger.info("Saved %d depth visualizations to %s", saved, output_dir)


def _camera_sort_key(path: Path):
    name = path.name
    return (0, int(name)) if name.isdigit() else (1, name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WHU-OMVS DA3MVS predict/test inference")
    parser.add_argument("--split", type=str, default="test", choices=["predict", "test"], help="predict or test")
    parser.add_argument("--dataset_root", type=str, default=str(_REPO_ROOT / "dataset" / "WHU-OMVS"))
    parser.add_argument("--output_path", type=str, default=str(_REPO_ROOT / "exp" / "whu-omvs" / "da3mvs_inference"))
    parser.add_argument("--checkpoint", type=str, default=str(_REPO_ROOT / "exp" / "whu-omvs" / "train_da3mvs" / "da3_large_adamvs_fusion_0526" / "checkpoints" / "best.pt"))
    parser.add_argument("--areas", type=str, nargs="*", default=["area2", "area3"])
    parser.add_argument("--camera_ids", type=str, nargs="*", default=["1", "2", "3", "4", "5"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--process_res", type=int, default=518)
    parser.add_argument("--process_res_method", type=str, default="square")
    parser.add_argument("--align_mode", type=str, default="median", choices=["none", "median"])
    parser.add_argument("--enable_viz", action="store_true", default=True)
    parser.add_argument("--viz_max_frames", type=int, default=-1, help="test split visualizes first 50 frames by shell default")
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    dataset_root = Path(_resolve_to_repo_path(args.dataset_root))
    output_path = Path(_resolve_to_repo_path(args.output_path))
    output_path.mkdir(parents=True, exist_ok=True)

    if args.device in ["cuda", "cpu"]:
        device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading DA3MVS checkpoint: %s", args.checkpoint)
    model = _load_da3mvs_model(_resolve_to_repo_path(args.checkpoint), device)

    split = args.split.lower()
    viz_root = output_path / "viz" / "depth_pngs"
    viz_root.mkdir(parents=True, exist_ok=True)
    metric_state = _init_metric_state()
    requested_camera_ids = None if not args.camera_ids else {str(camera_id) for camera_id in args.camera_ids}

    if split == "predict":
        split_root = dataset_root / "predict" / "Images"
        if not split_root.exists():
            raise FileNotFoundError(f"Predict split not found: {split_root}")
        cam_dirs = [p for p in split_root.iterdir() if p.is_dir() and (requested_camera_ids is None or p.name in requested_camera_ids)]
        for cam_dir in sorted(cam_dirs, key=_camera_sort_key):
            image_paths = list(_iter_image_paths(cam_dir))
            if len(image_paths) == 0:
                continue
            logger.info("Running predict viz for cam=%s (%d images)", cam_dir.name, len(image_paths))
            out_dir = viz_root / cam_dir.name
            _run_one_camera(
                model=model,
                image_paths=image_paths,
                output_dir=out_dir,
                device=device,
                process_res=args.process_res,
                process_res_method=args.process_res_method,
                batch_size=args.batch_size,
                viz_max_frames=args.viz_max_frames,
                metric_state=metric_state,
                split=split,
                dataset_root=dataset_root,
                area_name=None,
                camera_id=cam_dir.name,
            )
    else:
        for area_name in args.areas:
            area_root = dataset_root / "test" / area_name / "images"
            if not area_root.exists():
                logger.warning("Missing test area: %s", area_root)
                continue
            cam_dirs = [p for p in area_root.iterdir() if p.is_dir() and (requested_camera_ids is None or p.name in requested_camera_ids)]
            for cam_dir in sorted(cam_dirs, key=_camera_sort_key):
                image_paths = list(_iter_image_paths(cam_dir))
                if len(image_paths) == 0:
                    continue
                logger.info("Running test viz for area=%s cam=%s (%d images)", area_name, cam_dir.name, len(image_paths))
                out_dir = viz_root / area_name / cam_dir.name
                _run_one_camera(
                    model=model,
                    image_paths=image_paths,
                    output_dir=out_dir,
                    device=device,
                    process_res=args.process_res,
                    process_res_method=args.process_res_method,
                    batch_size=args.batch_size,
                    viz_max_frames=50 if args.viz_max_frames < 0 else args.viz_max_frames,
                    metric_state=metric_state,
                    split=split,
                    dataset_root=dataset_root,
                    area_name=area_name,
                    camera_id=cam_dir.name,
                )

    if metric_state["per_camera"]:
        if args.align_mode != "median":
            logger.warning("DA3MVS metrics are currently standardized to median alignment for comparability; requested align_mode=%s will be reflected only in metadata.", args.align_mode)
        _save_metric_summary(output_path, dataset_root, split, args.areas if split == "test" else [], metric_state, align_mode="median")

    logger.info("Done. Visualizations saved to: %s", viz_root)


if __name__ == "__main__":
    main()