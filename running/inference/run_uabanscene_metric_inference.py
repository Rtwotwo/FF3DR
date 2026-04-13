import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

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
from running.utils.config_utils import load_config
from uniception.models.utils.transformer_blocks import Mlp

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_DEPTH_EXTS = {".exr", ".npy", ".npz", ".pfm", ".png", ".tif", ".tiff"}


def _resolve_path(path_value: Optional[str]) -> Optional[str]:
	if path_value is None:
		return path_value
	path = Path(path_value)
	if path.is_absolute():
		return str(path)
	return str((_REPO_ROOT / path).resolve())


def _natural_key(text: str):
	return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", str(text))]


def _iter_files(root: Path, allowed_exts: set):
	for path in root.rglob("*"):
		if path.is_file() and path.suffix.lower() in allowed_exts:
			yield path


def _load_exr_depth(path: Path) -> np.ndarray:
	exr_file = OpenEXR.InputFile(str(path))
	header = exr_file.header()
	data_window = header["dataWindow"]
	width = data_window.max.x - data_window.min.x + 1
	height = data_window.max.y - data_window.min.y + 1
	channels = list(header.get("channels", {}).keys())
	preferred = ["Y", "R", "Z"]
	channel_name = None
	for name in preferred:
		if name in channels:
			channel_name = name
			break
	if channel_name is None:
		channel_name = channels[0]
	pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
	depth_bytes = exr_file.channel(channel_name, pixel_type)
	depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(height, width)
	return depth.copy()


def _load_pfm(path: Path) -> np.ndarray:
	with open(path, "rb") as handle:
		header = handle.readline().rstrip().decode("ascii")
		if header not in {"PF", "Pf"}:
			raise RuntimeError(f"[ERROR] Invalid PFM header: {header} for {path}")
		dims_line = handle.readline().decode("ascii").strip()
		while dims_line.startswith("#"):
			dims_line = handle.readline().decode("ascii").strip()
		width, height = map(int, dims_line.split())
		scale = float(handle.readline().decode("ascii").strip())
		endian = "<" if scale < 0 else ">"
		data = np.fromfile(handle, endian + "f")
		channels = 3 if header == "PF" else 1
		expected = width * height * channels
		if data.size != expected:
			raise RuntimeError(f"[ERROR] Invalid PFM size for {path}, expected={expected}, got={data.size}")
		if channels == 3:
			data = data.reshape((height, width, 3))[:, :, 0]
		else:
			data = data.reshape((height, width))
		data = np.flipud(data)
		return data.astype(np.float32)


def _load_depth(path: Path, depth_scale: float) -> np.ndarray:
	suffix = path.suffix.lower()
	if suffix == ".exr":
		return _load_exr_depth(path)
	if suffix == ".npy":
		return np.asarray(np.load(path), dtype=np.float32)
	if suffix == ".npz":
		npz_data = np.load(path)
		if "depth" in npz_data:
			return np.asarray(npz_data["depth"], dtype=np.float32)
		keys = list(npz_data.keys())
		if not keys:
			raise RuntimeError(f"[ERROR] Empty npz file: {path}")
		return np.asarray(npz_data[keys[0]], dtype=np.float32)
	if suffix == ".pfm":
		return _load_pfm(path)
	if suffix in {".png", ".tif", ".tiff"}:
		depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
		if depth is None:
			raise RuntimeError(f"[ERROR] Failed to read depth file: {path}")
		depth = depth.astype(np.float32)
		if depth_scale > 0:
			depth = depth / depth_scale
		return depth
	raise RuntimeError(f"[ERROR] Unsupported depth file format: {path}")


def _load_mask(path: Optional[Path], shape) -> np.ndarray:
	if path is None or not path.exists():
		return np.ones(shape, dtype=np.float32)
	mask = Image.open(path).convert("L")
	mask = np.asarray(mask, dtype=np.float32) / 255.0
	return (mask > 0.5).astype(np.float32)


def _prepare_prediction(pred_depth, gt_depth):
	pred_depth = np.asarray(pred_depth, dtype=np.float32)
	gt_depth = np.asarray(gt_depth, dtype=np.float32)
	pred_depth = np.squeeze(pred_depth)
	if pred_depth.ndim == 3:
		if pred_depth.shape[0] == 1:
			pred_depth = pred_depth[0]
		elif pred_depth.shape[-1] == 1:
			pred_depth = pred_depth[..., 0]
	if pred_depth.ndim != 2:
		raise RuntimeError(f"[ERROR] pred_depth must be 2D, got={pred_depth.shape}")
	gt_depth = np.squeeze(gt_depth)
	if gt_depth.ndim != 2:
		raise RuntimeError(f"[ERROR] gt_depth must be 2D, got={gt_depth.shape}")
	if pred_depth.shape != gt_depth.shape:
		pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
	return pred_depth


class _FileIndex:
	def __init__(self):
		self.by_rel = {}
		self.by_stem = defaultdict(list)

	def add(self, file_path: Path, rel_to_root: Path):
		rel_key = str(rel_to_root.with_suffix("")).replace("\\", "/")
		self.by_rel[rel_key] = file_path
		self.by_stem[file_path.stem].append(file_path)


class MetricInfer:
	"""Run depth metric inference for UrbanScene scenes."""

	def __init__(
		self,
		cfg,
		dataset_path,
		scenes,
		depth_root,
		mask_root,
		batch_size,
		align_mode,
		depth_scale,
	):
		self.cfg = cfg
		self.dataset_path = Path(dataset_path)
		self.scenes = scenes
		self.depth_root = Path(depth_root) if depth_root else None
		self.mask_root = Path(mask_root) if mask_root else None
		self.batch_size = max(1, int(batch_size))
		self.align_mode = align_mode
		self.depth_scale = float(depth_scale)
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
			return _resolve_path(weights[model_key][field])
		if fallback_flat_key is not None and fallback_flat_key in weights:
			return _resolve_path(weights[fallback_flat_key])
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

	def _resolve_scenes(self) -> List[str]:
		if self.scenes:
			return self.scenes
		scene_names = [path.name for path in self.dataset_path.iterdir() if path.is_dir()]
		return sorted(scene_names, key=_natural_key)

	def _collect_images(self, scene_dir: Path) -> List[Path]:
		image_paths = [path for path in scene_dir.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_EXTS]
		if image_paths:
			return sorted(image_paths, key=lambda path: _natural_key(path.name))
		recursive_images = list(_iter_files(scene_dir, _IMAGE_EXTS))
		return sorted(recursive_images, key=lambda path: _natural_key(path.relative_to(scene_dir).as_posix()))

	def _build_index(self, roots: List[Path]) -> _FileIndex:
		index = _FileIndex()
		for root in roots:
			if not root.is_dir():
				continue
			for file_path in _iter_files(root, _DEPTH_EXTS):
				try:
					rel = file_path.relative_to(root)
				except ValueError:
					rel = Path(file_path.name)
				index.add(file_path, rel)
		return index

	def _candidate_depth_roots(self, scene_name: str, scene_dir: Path) -> List[Path]:
		candidates = []
		if self.depth_root is not None:
			candidates.extend([self.depth_root / scene_name, self.depth_root, self.depth_root / "depth" / scene_name])
		candidates.extend(
			[
				scene_dir / "depth",
				scene_dir / "depths",
				self.dataset_path / "depth" / scene_name,
				self.dataset_path / "depths" / scene_name,
				self.dataset_path / "depth_gt" / scene_name,
				self.dataset_path / scene_name / "depth",
				self.dataset_path / scene_name / "depths",
			]
		)
		seen = set()
		ordered = []
		for item in candidates:
			key = str(item.resolve()) if item.exists() else str(item)
			if key in seen:
				continue
			seen.add(key)
			ordered.append(item)
		return ordered

	def _candidate_mask_roots(self, scene_name: str, scene_dir: Path) -> List[Path]:
		candidates = []
		if self.mask_root is not None:
			candidates.extend([self.mask_root / scene_name, self.mask_root, self.mask_root / "mask" / scene_name])
		candidates.extend(
			[
				scene_dir / "mask",
				scene_dir / "masks",
				self.dataset_path / "mask" / scene_name,
				self.dataset_path / "masks" / scene_name,
				self.dataset_path / scene_name / "mask",
				self.dataset_path / scene_name / "masks",
			]
		)
		seen = set()
		ordered = []
		for item in candidates:
			key = str(item.resolve()) if item.exists() else str(item)
			if key in seen:
				continue
			seen.add(key)
			ordered.append(item)
		return ordered

	def _match_from_index(self, index: _FileIndex, rel_noext: str, stem: str) -> Optional[Path]:
		if rel_noext in index.by_rel:
			return index.by_rel[rel_noext]
		candidates = index.by_stem.get(stem, [])
		if len(candidates) == 1:
			return candidates[0]
		if len(candidates) > 1:
			return sorted(candidates, key=lambda path: _natural_key(path.as_posix()))[0]
		return None

	def _collect_samples_for_scene(self, scene_name: str) -> List[Dict]:
		scene_dir = self.dataset_path / scene_name
		if not scene_dir.is_dir():
			logger.warning("[WARN] Skip missing scene folder: %s", scene_dir)
			return []

		image_paths = self._collect_images(scene_dir)
		if not image_paths:
			logger.warning("[WARN] No images found in scene: %s", scene_dir)
			return []

		depth_index = self._build_index(self._candidate_depth_roots(scene_name, scene_dir))
		mask_index = self._build_index(self._candidate_mask_roots(scene_name, scene_dir))

		samples = []
		for image_path in image_paths:
			rel = image_path.relative_to(scene_dir)
			rel_noext = str(rel.with_suffix("")).replace("\\", "/")
			depth_path = self._match_from_index(depth_index, rel_noext, image_path.stem)
			if depth_path is None:
				continue
			mask_path = self._match_from_index(mask_index, rel_noext, image_path.stem)
			samples.append(
				{
					"image_path": image_path,
					"depth_path": depth_path,
					"mask_path": mask_path,
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
			depth = np.asarray(depth, dtype=np.float32)
			depth = np.squeeze(depth)
			if depth.ndim == 3:
				if depth.shape[0] == 1:
					depth = depth[0]
				elif depth.shape[-1] == 1:
					depth = depth[..., 0]
			if depth.ndim != 2:
				raise RuntimeError(f"[ERROR] Unexpected mapanything depth shape for {image_path}: {depth.shape}")
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

	def run(self):
		scenes = self._resolve_scenes()
		per_scene_results = {}
		overall_accumulator = DepthMetricAccumulator(align_mode=self.align_mode)

		for scene_name in scenes:
			samples = self._collect_samples_for_scene(scene_name)
			if not samples:
				logger.warning("[WARN] No matched image-depth samples found for scene %s", scene_name)
				continue

			scene_accumulator = DepthMetricAccumulator(align_mode=self.align_mode)
			logger.info("[INFO] Evaluating scene=%s samples=%d", scene_name, len(samples))

			for start_idx in range(0, len(samples), self.batch_size):
				batch_samples = samples[start_idx : start_idx + self.batch_size]
				batch_images = [sample["image_path"] for sample in batch_samples]
				batch_predictions = self._predict_batch(batch_images)

				for batch_index, sample in enumerate(batch_samples):
					pred_depth = batch_predictions[batch_index]
					gt_depth = _load_depth(sample["depth_path"], depth_scale=self.depth_scale)
					mask = _load_mask(sample["mask_path"], np.squeeze(gt_depth).shape)
					pred_depth = _prepare_prediction(pred_depth, gt_depth)
					scene_accumulator.update(pred_depth, gt_depth, mask)

			scene_result = scene_accumulator.finalize()
			per_scene_results[scene_name] = scene_result
			overall_accumulator.merge(scene_accumulator)

		overall_result = overall_accumulator.finalize()
		return {
			"model_name": self.model_name,
			"dataset": "UrbanScene",
			"dataset_path": str(self.dataset_path),
			"align_mode": self.align_mode,
			"scenes": per_scene_results,
			"overall": overall_result,
		}


def _format_metric(value):
	if value is None or (isinstance(value, float) and not np.isfinite(value)):
		return "N/A"
	return f"{value:.6f}"


def _print_results(results):
	overall = results["overall"]
	print("=" * 92)
	print("UrbanScene Depth Metrics")
	print("=" * 92)
	print(f"model={results['model_name']} align={results['align_mode']}")
	print(f"valid_pixels={overall['valid_count']}")
	print("-" * 92)
	headers = ["abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog", "delta1", "delta2", "delta3"]
	print(" | ".join(f"{header:>10}" for header in headers))
	print(" | ".join(f"{_format_metric(overall[header]):>10}" for header in headers))
	print("-" * 92)
	for scene_name, scene_result in results["scenes"].items():
		values = " | ".join(f"{_format_metric(scene_result[header]):>10}" for header in headers)
		print(f"{scene_name:>18}: {values}")


def _load_run_arg_defaults(yaml_path):
	if yaml_path is None or not os.path.isfile(yaml_path):
		return {}
	cfg = load_config(yaml_path)
	if not isinstance(cfg, dict):
		return {}
	defaults = cfg.get("RunArgs", cfg)
	return defaults if isinstance(defaults, dict) else {}


def _build_parser(run_defaults):
	parser = argparse.ArgumentParser(description="UrbanScene metric inference")
	parser.add_argument(
		"--run_args_yaml",
		type=str,
		default=str(_REPO_ROOT / "configs" / "run_urbanscene_metric_inference.yaml"),
		help="YAML file with default startup arguments",
	)
	parser.add_argument(
		"--config_path",
		type=str,
		default=run_defaults.get("config_path", str(_REPO_ROOT / "configs" / "base_config.yaml")),
		help="Path to the model config file.",
	)
	parser.add_argument(
		"--dataset_path",
		type=str,
		default=run_defaults.get("dataset_path", str(_REPO_ROOT / "dataset" / "UrbanScene")),
		help="Path to UrbanScene root.",
	)
	parser.add_argument(
		"--scenes",
		nargs="*",
		default=run_defaults.get("scenes", None),
		help="Optional scene list. Empty means all folders under dataset path.",
	)
	parser.add_argument(
		"--depth_root",
		type=str,
		default=run_defaults.get("depth_root", None),
		help="Optional depth root override.",
	)
	parser.add_argument(
		"--mask_root",
		type=str,
		default=run_defaults.get("mask_root", None),
		help="Optional mask root override.",
	)
	parser.add_argument(
		"--depth_scale",
		type=float,
		default=float(run_defaults.get("depth_scale", 1000.0)),
		help="Depth divisor for integer depth images (png/tiff).",
	)
	parser.add_argument("--batch_size", type=int, default=int(run_defaults.get("batch_size", 8)))
	parser.add_argument(
		"--align_mode",
		type=str,
		default=run_defaults.get("align_mode", "median"),
		choices=["none", "median"],
	)
	parser.add_argument(
		"--output_path",
		type=str,
		default=run_defaults.get("output_path", str(_REPO_ROOT / "exp" / "urbanscene" / "metric_eval")),
		help="Directory for saving metrics json.",
	)
	parser.add_argument(
		"--model_name",
		type=str,
		default=run_defaults.get("model_name", None),
		choices=["depthanything3", "mapanything", "pi3", "vggt"],
		help="Model name override.",
	)
	parser.add_argument("--log_level", type=str, default=run_defaults.get("log_level", "INFO"))
	return parser


def main():
	pre_parser = argparse.ArgumentParser(add_help=False)
	pre_parser.add_argument(
		"--run_args_yaml",
		type=str,
		default=str(_REPO_ROOT / "configs" / "run_urbanscene_metric_inference.yaml"),
	)
	pre_args, _ = pre_parser.parse_known_args()
	defaults = _load_run_arg_defaults(_resolve_path(pre_args.run_args_yaml))

	parser = _build_parser(defaults)
	args = parser.parse_args()
	logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

	config_path = _resolve_path(args.config_path)
	dataset_path = _resolve_path(args.dataset_path)
	output_path = Path(_resolve_path(args.output_path))
	output_path.mkdir(parents=True, exist_ok=True)

	cfg = load_config(config_path)
	if args.model_name is not None:
		if "Model" not in cfg or not isinstance(cfg["Model"], dict):
			cfg["Model"] = {}
		cfg["Model"]["name"] = args.model_name

	runner = MetricInfer(
		cfg=cfg,
		dataset_path=dataset_path,
		scenes=args.scenes,
		depth_root=_resolve_path(args.depth_root),
		mask_root=_resolve_path(args.mask_root),
		batch_size=args.batch_size,
		align_mode=args.align_mode,
		depth_scale=args.depth_scale,
	)
	results = runner.run()
	_print_results(results)

	output_file = output_path / f"urbanscene_{results['model_name']}_metrics.json"
	with open(output_file, "w", encoding="utf-8") as file_handle:
		json.dump(results, file_handle, indent=2)
	logger.info("[INFO] Metrics saved to %s", output_file)


if __name__ == "__main__":
	main()
