from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import cv2
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
from running.utils.viz_utils import depth_to_color

logger = logging.getLogger(__name__)


def _resolve_to_repo_path(path_value: str | None) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((_REPO_ROOT / path).resolve())


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


def _preprocess_image(image_path: str, target_size: int, process_res_method: str) -> torch.Tensor:
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
    return torch.from_numpy(image_array).permute(2, 0, 1).contiguous()


def _as_depth_map(value: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
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


def _iter_image_paths(root_dir: Path) -> Iterable[Path]:
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        for path in sorted(root_dir.glob(pattern)):
            yield path


def _camera_sort_key(path: Path):
    name = path.name
    return (0, int(name)) if name.isdigit() else (1, name)


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


def _write_pfm(file_path: Path, image: np.ndarray) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        color = False
    elif image.ndim == 3 and image.shape[2] in (1, 3):
        color = image.shape[2] == 3
        if image.shape[2] == 1:
            image = image[:, :, 0]
    else:
        raise ValueError(f"Unsupported PFM shape: {image.shape}")

    image = np.flipud(image)
    with open(file_path, "wb") as fh:
        fh.write(b"PF\n" if color else b"Pf\n")
        fh.write(f"{image.shape[1]} {image.shape[0]}\n".encode("ascii"))
        endian_scale = -1.0 if image.dtype.byteorder == "<" or (image.dtype.byteorder == "=" and np.little_endian) else 1.0
        fh.write(f"{endian_scale}\n".encode("ascii"))
        fh.write(image.astype(np.float32).tobytes())


def _run_one_camera(
    model,
    image_paths: list[Path],
    output_dir: Path,
    device: torch.device,
    process_res: int,
    process_res_method: str,
    batch_size: int,
    viz_max_frames: int,
) -> None:
    if not image_paths:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch_tensors = [_preprocess_image(str(image_path), process_res, process_res_method) for image_path in batch_paths]
        if not batch_tensors:
            continue

        images = torch.stack(batch_tensors, dim=0).to(device)
        images_input = images.unsqueeze(1)

        with torch.no_grad():
            autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            if torch.cuda.is_available():
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    pred_depth, pred_metric, _, _, _ = model(images_input)
            else:
                pred_depth, pred_metric, _, _, _ = model(images_input)

        depth_maps = _as_depth_map(pred_metric if pred_metric is not None else pred_depth)
        if depth_maps is None:
            logger.warning("No depth predictions returned for batch starting at %d", start)
            continue
        if depth_maps.ndim == 4:
            logger.warning("DA3MVS depth output has extra channel dimension %s; using the first channel for visualization.", depth_maps.shape)
            depth_maps = depth_maps[:, 0]
        if depth_maps.ndim == 2:
            depth_maps = depth_maps[None, ...]

        for idx, image_path in enumerate(batch_paths):
            if viz_max_frames > 0 and saved >= viz_max_frames:
                break
            _write_pfm(output_dir / f"{image_path.stem}_init.pfm", depth_maps[idx])
            saved += 1

    logger.info("Saved %d depth visualizations to %s", saved, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WHU-OMVS DA3MVS predict/test visualization")
    parser.add_argument("--split", type=str, default="test", choices=["predict", "test"], help="predict or test")
    parser.add_argument("--dataset_root", type=str, default=str(_REPO_ROOT / "dataset" / "WHU-OMVS"))
    parser.add_argument("--output_path", type=str, default=str(_REPO_ROOT / "exp" / "whu-omvs" / "metric_da3mvs_whuomvs_predict"))
    parser.add_argument("--checkpoint", type=str, default=str(_REPO_ROOT / "exp" / "whu-omvs" / "train_da3mvs" / "da3_large_adamvs_fusion_0526" / "checkpoints" / "best.pt"))
    parser.add_argument("--areas", type=str, nargs="*", default=["area2", "area3"])
    parser.add_argument("--camera_ids", type=str, nargs="*", default=["1", "2", "3", "4", "5"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--process_res", type=int, default=518)
    parser.add_argument("--process_res_method", type=str, default="square")
    parser.add_argument("--viz_max_frames", type=int, default=-1, help="test split visualizes first 50 frames by shell default")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    dataset_root = Path(_resolve_to_repo_path(args.dataset_root))
    output_path = Path(_resolve_to_repo_path(args.output_path))
    output_path.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading DA3MVS checkpoint: %s", args.checkpoint)
    model = _load_da3mvs_model(_resolve_to_repo_path(args.checkpoint), device)

    split = args.split.lower()
    raw_root = output_path / "da3mvs_output"
    raw_root.mkdir(parents=True, exist_ok=True)
    viz_root = output_path / "viz" / "depth_pngs"
    viz_root.mkdir(parents=True, exist_ok=True)
    requested_camera_ids = None if not args.camera_ids else {str(camera_id) for camera_id in args.camera_ids}

    if split == "predict":
        split_root = dataset_root / "predict" / "Images"
        if not split_root.exists():
            raise FileNotFoundError(f"Predict split not found: {split_root}")
        cam_dirs = [p for p in split_root.iterdir() if p.is_dir() and (requested_camera_ids is None or p.name in requested_camera_ids)]
        for cam_dir in sorted(cam_dirs, key=_camera_sort_key):
            image_paths = list(_iter_image_paths(cam_dir))
            if not image_paths:
                continue
            logger.info("Running predict viz for cam=%s (%d images)", cam_dir.name, len(image_paths))
            _run_one_camera(
                model=model,
                image_paths=image_paths,
                output_dir=raw_root / cam_dir.name,
                device=device,
                process_res=args.process_res,
                process_res_method=args.process_res_method,
                batch_size=args.batch_size,
                viz_max_frames=args.viz_max_frames,
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
                if not image_paths:
                    continue
                logger.info("Running test viz for area=%s cam=%s (%d images)", area_name, cam_dir.name, len(image_paths))
                _run_one_camera(
                    model=model,
                    image_paths=image_paths,
                    output_dir=raw_root / area_name / cam_dir.name,
                    device=device,
                    process_res=args.process_res,
                    process_res_method=args.process_res_method,
                    batch_size=args.batch_size,
                    viz_max_frames=50 if args.viz_max_frames < 0 else args.viz_max_frames,
                )

    from running.training.datasets_adamvs.data_io import read_pfm

    for root, _, files in os.walk(raw_root):
        for fn in files:
            if not fn.endswith("_init.pfm"):
                continue
            src = Path(root) / fn
            rel = src.relative_to(raw_root)
            dst_dir = viz_root / rel.parent
            dst_dir.mkdir(parents=True, exist_ok=True)
            try:
                depth, _ = read_pfm(str(src))
            except Exception as exc:
                logger.warning("Failed to read PFM %s: %s", src, exc)
                continue
            if depth.ndim == 3:
                depth = depth[..., 0]
            color = depth_to_color(depth)
            cv2.imwrite(str(dst_dir / f"{src.stem.replace('_init', '')}.png"), color)

    logger.info("Done. Visualizations saved to: %s", viz_root)


if __name__ == "__main__":
    main()