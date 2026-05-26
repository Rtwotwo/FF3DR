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

    if split == "predict":
        split_root = dataset_root / "predict" / "Images"
        if not split_root.exists():
            raise FileNotFoundError(f"Predict split not found: {split_root}")
        cam_dirs = [p for p in split_root.iterdir() if p.is_dir()]
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
            )
    else:
        for area_name in args.areas:
            area_root = dataset_root / "test" / area_name / "images"
            if not area_root.exists():
                logger.warning("Missing test area: %s", area_root)
                continue
            cam_dirs = [p for p in area_root.iterdir() if p.is_dir()]
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
                )

    logger.info("Done. Visualizations saved to: %s", viz_root)


if __name__ == "__main__":
    main()