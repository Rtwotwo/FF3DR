import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
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

from running.utils.config_utils import load_config
from models.g3splat.config import load_typed_root_config
from models.g3splat.dataset.data_module import get_data_shim
from models.g3splat.model.decoder import get_decoder
from models.g3splat.model.encoder import get_encoder
from models.g3splat.misc.cam_utils import get_pnp_pose
from models.g3splat.visualization.camera_trajectory.interpolation import (
	interpolate_extrinsics,
	interpolate_intrinsics,
)


logger = logging.getLogger(__name__)


def _write_point_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
	"""Write XYZRGB point cloud as binary little-endian PLY."""
	if points is None or colors is None or len(points) == 0:
		return
	points = np.asarray(points, dtype=np.float32)
	colors = np.asarray(colors, dtype=np.uint8)
	if points.ndim != 2 or points.shape[1] != 3:
		raise ValueError(f"Invalid points shape for ply export: {points.shape}")
	if colors.ndim != 2 or colors.shape[1] != 3:
		raise ValueError(f"Invalid colors shape for ply export: {colors.shape}")
	n = min(points.shape[0], colors.shape[0])
	points = points[:n]
	colors = colors[:n]

	vertex = np.empty(
		n,
		dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
	)
	vertex["x"] = points[:, 0]
	vertex["y"] = points[:, 1]
	vertex["z"] = points[:, 2]
	vertex["red"] = colors[:, 0]
	vertex["green"] = colors[:, 1]
	vertex["blue"] = colors[:, 2]

	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "wb") as f:
		header = (
			"ply\n"
			"format binary_little_endian 1.0\n"
			f"element vertex {n}\n"
			"property float x\n"
			"property float y\n"
			"property float z\n"
			"property uchar red\n"
			"property uchar green\n"
			"property uchar blue\n"
			"end_header\n"
		)
		f.write(header.encode("ascii"))
		vertex.tofile(f)


def _transform_points_se3(points: np.ndarray, pose_c2w: np.ndarray) -> np.ndarray:
	"""Apply c2w SE3 pose to Nx3 points."""
	if points is None or points.size == 0:
		return points
	if pose_c2w is None:
		return points
	r = pose_c2w[:3, :3].astype(np.float32)
	t = pose_c2w[:3, 3].astype(np.float32)
	return (points @ r.T) + t[None, :]


def _resolve_to_repo_path(path_value: str) -> str:
	p = Path(path_value)
	if p.is_absolute():
		return str(p)
	return str((_REPO_ROOT / p).resolve())


def _load_run_arg_defaults(yaml_path: str) -> Dict:
	if yaml_path is None or not os.path.isfile(yaml_path):
		return {}
	cfg = load_config(yaml_path)
	if not isinstance(cfg, dict):
		return {}
	defaults = cfg.get("RunArgs", cfg)
	return defaults if isinstance(defaults, dict) else {}


def _camera_sort_key(path: str):
	name = os.path.basename(path)
	return (0, int(name)) if name.isdigit() else (1, name)


def _collect_scene_images(area_path: str, anchor_camera: int = 0) -> List[str]:
	if not os.path.isdir(area_path):
		raise RuntimeError(f"[ERROR] area_path not found: {area_path}")

	files = [
		os.path.join(area_path, f)
		for f in os.listdir(area_path)
		if os.path.isfile(os.path.join(area_path, f)) and f.lower().endswith((".png", ".jpg", ".jpeg"))
	]
	if len(files) > 0:
		return sorted(files)

	camera_dirs = [
		os.path.join(area_path, d)
		for d in os.listdir(area_path)
		if os.path.isdir(os.path.join(area_path, d))
	]
	camera_dirs = sorted(camera_dirs, key=_camera_sort_key)
	if len(camera_dirs) == 0:
		raise RuntimeError(f"[ERROR] No images/camera folders under: {area_path}")
	if not (0 <= anchor_camera < len(camera_dirs)):
		raise ValueError(
			f"[SETTING ERROR] anchor_camera={anchor_camera} out of range [0,{len(camera_dirs)-1}]"
		)

	cam_dir = camera_dirs[anchor_camera]
	cam_files = [
		os.path.join(cam_dir, f)
		for f in os.listdir(cam_dir)
		if os.path.isfile(os.path.join(cam_dir, f)) and f.lower().endswith((".png", ".jpg", ".jpeg"))
	]
	cam_files = sorted(cam_files)
	if len(cam_files) == 0:
		raise RuntimeError(f"[ERROR] anchor camera folder has no images: {cam_dir}")
	logger.info("[INFO] Multi-camera input detected, using anchor camera folder: %s", cam_dir)
	return cam_files


def _read_image(path: str, size: int) -> torch.Tensor:
	img = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
	arr = np.asarray(img, dtype=np.float32) / 255.0
	ten = torch.from_numpy(arr).permute(2, 0, 1)
	return ten


class ChunkedUrbanSceneNVS:
	def __init__(self, args: argparse.Namespace):
		self.args = args
		self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
		self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

		self.area_path = _resolve_to_repo_path(args.area_path)
		self.output_path = _resolve_to_repo_path(args.output_path)
		self.g3splat_config_dir = _resolve_to_repo_path(args.g3splat_config_dir)
		self.checkpoint = _resolve_to_repo_path(args.checkpoint)
		os.makedirs(self.output_path, exist_ok=True)

		self.save_ply = bool(int(args.save_ply))
		self.save_pair_ply = bool(int(args.save_pair_ply))
		self.save_merged_ply = bool(int(args.save_merged_ply))
		self.ply_dir = os.path.join(self.output_path, "ply")
		self.ply_pair_dir = os.path.join(self.ply_dir, "pairs")
		if self.save_ply:
			os.makedirs(self.ply_dir, exist_ok=True)
			if self.save_pair_ply:
				os.makedirs(self.ply_pair_dir, exist_ok=True)

		if not os.path.isdir(self.g3splat_config_dir):
			raise RuntimeError(f"[ERROR] g3splat config dir not found: {self.g3splat_config_dir}")
		if not os.path.isfile(self.checkpoint):
			raise RuntimeError(f"[ERROR] checkpoint not found: {self.checkpoint}")

		self.image_paths = _collect_scene_images(self.area_path, args.anchor_camera)
		if len(self.image_paths) < 2:
			raise RuntimeError("[ERROR] Need at least 2 images to perform NVS interpolation")

		self.encoder, self.decoder, self.data_shim = self._load_model()

	def _load_model(self):
		GlobalHydra.instance().clear()
		with initialize_config_dir(version_base=None, config_dir=self.g3splat_config_dir):
			cfg_dict: DictConfig = compose(
				config_name="main",
				overrides=[
					f"+experiment={self.args.experiment}",
					f"model.encoder.gaussian_adapter.gaussian_type={self.args.gaussian_type}",
					"mode=test",
				],
			)
		cfg = load_typed_root_config(cfg_dict)

		encoder, _ = get_encoder(cfg.model.encoder)
		decoder = get_decoder(cfg.model.decoder)

		ckpt = torch.load(self.checkpoint, map_location="cpu")
		state_dict = ckpt.get("state_dict", ckpt)
		encoder_state = {k[8:]: v for k, v in state_dict.items() if k.startswith("encoder.")}
		missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
		if len(unexpected) > 0:
			logger.warning("[WARN] Unexpected encoder keys: %d", len(unexpected))
		if len(missing) > 0:
			logger.warning("[WARN] Missing encoder keys: %d", len(missing))

		encoder = encoder.to(self.device).eval()
		decoder = decoder.to(self.device).eval()
		data_shim = get_data_shim(encoder)
		return encoder, decoder, data_shim

	def _make_intrinsics(self) -> torch.Tensor:
		fx = float(self.args.fx)
		fy = float(self.args.fy)
		cx = float(self.args.cx)
		cy = float(self.args.cy)
		k = torch.tensor(
			[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
			dtype=torch.float32,
			device=self.device,
		)
		return k

	def _make_pair_batch(self, img_a: torch.Tensor, img_b: torch.Tensor) -> Dict:
		images = torch.stack([img_a, img_b], dim=0).unsqueeze(0).to(self.device)
		k = self._make_intrinsics().unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)
		ext = torch.eye(4, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)
		near = torch.tensor([[self.args.near, self.args.near]], dtype=torch.float32, device=self.device)
		far = torch.tensor([[self.args.far, self.args.far]], dtype=torch.float32, device=self.device)
		batch = {
			"context": {
				"image": images,
				"intrinsics": k,
				"extrinsics": ext,
				"near": near,
				"far": far,
			},
			"target": {
				"image": images,
				"intrinsics": k,
				"extrinsics": ext,
				"near": near,
				"far": far,
			},
		}
		return self.data_shim(batch)

	def _estimate_pair_pose(self, vis_dump: Dict, intrinsics: torch.Tensor, h: int, w: int) -> torch.Tensor:
		pose_1 = torch.eye(4, dtype=torch.float32, device=self.device)

		pts3d = vis_dump["means"][0, 1].squeeze(-2).reshape(-1, 3)
		opacity = vis_dump["opacities"][0, 1].squeeze(-1).squeeze(-1).reshape(-1)
		try:
			pose_2 = get_pnp_pose(
				pts3d,
				opacity,
				intrinsics[0, 1],
				h,
				w,
				opacity_threshold=float(self.args.pnp_opacity_threshold),
				use_ransac=True,
			).to(self.device)
		except Exception as ex:
			logger.warning("[WARN] PnP failed, fallback translation. reason=%s", str(ex))
			pose_2 = torch.eye(4, dtype=torch.float32, device=self.device)
			pose_2[0, 3] = float(self.args.fallback_baseline)

		return torch.stack([pose_1, pose_2], dim=0)

	def _extract_pair_points(
		self,
		vis_dump: Dict,
		img_a: torch.Tensor,
		img_b: torch.Tensor,
	) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
		"""Extract sampled XYZRGB points from current pair for PLY export."""
		if not self.save_ply:
			return None, None

		try:
			pts = vis_dump["means"][0].squeeze(-2)  # [2, H, W, 3]
			opacity = vis_dump["opacities"][0].squeeze(-1).squeeze(-1)  # [2, H, W]
		except Exception:
			return None, None

		if pts.ndim != 4 or opacity.ndim != 3:
			return None, None

		colors = torch.stack([img_a, img_b], dim=0).permute(0, 2, 3, 1).contiguous()  # [2, H, W, 3]
		mask = opacity > float(self.args.ply_opacity_threshold)
		mask_cpu = mask.detach().cpu()

		pts_np = pts[mask].detach().cpu().numpy().astype(np.float32)
		if pts_np.shape[0] == 0:
			return None, None
		colors_np = (colors[mask_cpu].detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)

		max_pp = int(self.args.max_points_per_pair)
		if max_pp > 0 and pts_np.shape[0] > max_pp:
			idx = np.random.choice(pts_np.shape[0], size=max_pp, replace=False)
			pts_np = pts_np[idx]
			colors_np = colors_np[idx]

		return pts_np, colors_np

	@torch.no_grad()
	def _render_pair_frames(
		self,
		img_a_path: str,
		img_b_path: str,
	) -> Tuple[List[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
		img_a = _read_image(img_a_path, self.args.image_size)
		img_b = _read_image(img_b_path, self.args.image_size)
		h, w = img_a.shape[1], img_a.shape[2]

		batch = self._make_pair_batch(img_a, img_b)
		vis_dump = {}
		with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=self.device.type == "cuda"):
			gaussians = self.encoder(batch["context"], global_step=0, visualization_dump=vis_dump)

		ply_points, ply_colors = self._extract_pair_points(vis_dump, img_a, img_b)

		poses = self._estimate_pair_pose(vis_dump, batch["context"]["intrinsics"], h, w)

		n = int(self.args.interp_frames_per_pair)
		t = torch.linspace(0.0, 1.0, n, dtype=torch.float32, device=self.device)
		t = (torch.cos(torch.pi * (t + 1.0)) + 1.0) / 2.0

		ext_interp = interpolate_extrinsics(poses[0], poses[1], t)
		int_interp = interpolate_intrinsics(
			batch["context"]["intrinsics"][0, 0],
			batch["context"]["intrinsics"][0, 1],
			t,
		)

		near = torch.full((1, n), float(self.args.near), dtype=torch.float32, device=self.device)
		far = torch.full((1, n), float(self.args.far), dtype=torch.float32, device=self.device)
		decoder_type = "3D" if self.args.gaussian_type.lower() == "3d" else "2D"

		frames: List[np.ndarray] = []
		render_bs = max(1, int(self.args.render_batch_size))
		for s in range(0, n, render_bs):
			e = min(s + render_bs, n)
			out = self.decoder.forward(
				gaussians,
				ext_interp[s:e].unsqueeze(0),
				int_interp[s:e].unsqueeze(0),
				near[:, s:e],
				far[:, s:e],
				(h, w),
				depth_mode="depth",
				decoder_type=decoder_type,
			)
			rgb = out.color[0]
			for i in range(rgb.shape[0]):
				frame = (rgb[i].clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
				frames.append(frame)

			if self.device.type == "cuda":
				torch.cuda.empty_cache()

		pose_2_np = poses[1].detach().cpu().numpy().astype(np.float32)
		return frames, ply_points, ply_colors, pose_2_np

	def _build_pair_chunks(self) -> List[Tuple[int, int]]:
		total_pairs = len(self.image_paths) - 1
		if total_pairs <= 0:
			return []
		chunk = int(self.args.pair_chunk_size)
		overlap = int(self.args.pair_overlap)
		if chunk <= 0:
			return [(0, total_pairs)]
		if overlap >= chunk:
			raise ValueError(f"[SETTING ERROR] pair_overlap={overlap} must be < pair_chunk_size={chunk}")
		step = chunk - overlap
		spans: List[Tuple[int, int]] = []
		s = 0
		while s < total_pairs:
			e = min(s + chunk, total_pairs)
			spans.append((s, e))
			if e == total_pairs:
				break
			s += step
		return spans

	def run(self):
		scene_name = os.path.basename(os.path.normpath(self.area_path))
		out_video = os.path.join(self.output_path, f"{scene_name}_g3splat_nvs.mp4")
		os.makedirs(self.output_path, exist_ok=True)

		all_frames: List[np.ndarray] = []
		all_ply_points: List[np.ndarray] = []
		all_ply_colors: List[np.ndarray] = []
		total_ply_points = 0
		seen_pairs = set()
		global_pose_cam0 = np.eye(4, dtype=np.float32)
		pair_chunks = self._build_pair_chunks()
		if self.args.max_pair_chunks > 0:
			pair_chunks = pair_chunks[: int(self.args.max_pair_chunks)]

		logger.info(
			"[INFO] Start NVS: scene=%s, images=%d, pairs=%d, pair_chunks=%d",
			scene_name,
			len(self.image_paths),
			len(self.image_paths) - 1,
			len(pair_chunks),
		)

		for ci, (ps, pe) in enumerate(pair_chunks):
			logger.info("[INFO] Pair chunk %d/%d: [%d, %d)", ci + 1, len(pair_chunks), ps, pe)
			for pidx in range(ps, pe):
				if pidx in seen_pairs:
					continue
				seen_pairs.add(pidx)
				frames, pair_points, pair_colors, pose_2 = self._render_pair_frames(
					self.image_paths[pidx], self.image_paths[pidx + 1]
				)
				if len(all_frames) > 0 and len(frames) > 0:
					all_frames.extend(frames[1:])
				else:
					all_frames.extend(frames)

				if self.save_ply and pair_points is not None and pair_colors is not None:
					if bool(int(self.args.ply_use_global_pose_alignment)):
						pair_points = _transform_points_se3(pair_points, global_pose_cam0)

					if self.save_pair_ply:
						pair_path = os.path.join(self.ply_pair_dir, f"pair_{pidx:06d}.ply")
						_write_point_ply(pair_path, pair_points, pair_colors)

					if self.save_merged_ply:
						all_ply_points.append(pair_points)
						all_ply_colors.append(pair_colors)
						total_ply_points += int(pair_points.shape[0])

				# Chain c2w pose: next pair's cam0 should equal current pair's cam1 in global frame.
				global_pose_cam0 = (global_pose_cam0 @ pose_2).astype(np.float32)

		if len(all_frames) == 0:
			raise RuntimeError("[ERROR] No rendered frames produced")

		h, w, _ = all_frames[0].shape
		writer = cv2.VideoWriter(
			out_video,
			cv2.VideoWriter_fourcc(*"mp4v"),
			float(self.args.fps),
			(w, h),
		)
		for frame in all_frames:
			writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
		writer.release()

		if self.save_ply and self.save_merged_ply and len(all_ply_points) > 0:
			points = np.concatenate(all_ply_points, axis=0)
			colors = np.concatenate(all_ply_colors, axis=0)

			max_total = int(self.args.max_total_ply_points)
			if max_total > 0 and points.shape[0] > max_total:
				idx = np.random.choice(points.shape[0], size=max_total, replace=False)
				points = points[idx]
				colors = colors[idx]

			merged_ply_path = os.path.join(self.ply_dir, f"{scene_name}_g3splat_nvs_points.ply")
			_write_point_ply(merged_ply_path, points, colors)
			logger.info(
				"[INFO] Saved merged NVS PLY: %s (points=%d, pre_merge_points=%d)",
				merged_ply_path,
				int(points.shape[0]),
				total_ply_points,
			)

		logger.info("[INFO] Saved chunk-stitched NVS video: %s", out_video)
		logger.info("[INFO] Total output frames: %d", len(all_frames))


def _build_parser(run_defaults: Dict) -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Chunked UrbanScene NVS inference with g3splat")
	parser.add_argument(
		"--run_args_yaml",
		type=str,
		default=str(_REPO_ROOT / "configs" / "run_urbanscene_nvs_inference.yaml"),
		help="YAML file with default startup arguments",
	)
	parser.add_argument("--area_path", type=str, default=run_defaults.get("area_path", "./dataset/UrbanScene/ArtSci"))
	parser.add_argument("--output_path", type=str, default=run_defaults.get("output_path", "./exp/urbanscene_nvs"))
	parser.add_argument("--g3splat_config_dir", type=str, default=run_defaults.get("g3splat_config_dir", "./clones/g3splat/config"))
	parser.add_argument("--checkpoint", type=str, default=run_defaults.get("checkpoint", "./weights/g3splat/g3splat_mast3r_3dgs_align_orient_re10k.ckpt"))
	parser.add_argument("--experiment", type=str, default=run_defaults.get("experiment", "re10k_align_orient"))
	parser.add_argument("--gaussian_type", type=str, default=run_defaults.get("gaussian_type", "3d"), help="2d or 3d")
	parser.add_argument("--device", type=str, default=run_defaults.get("device", "cuda"))
	parser.add_argument("--anchor_camera", type=int, default=int(run_defaults.get("anchor_camera", 0)))
	parser.add_argument("--image_size", type=int, default=int(run_defaults.get("image_size", 256)))
	parser.add_argument("--interp_frames_per_pair", type=int, default=int(run_defaults.get("interp_frames_per_pair", 16)))
	parser.add_argument("--render_batch_size", type=int, default=int(run_defaults.get("render_batch_size", 4)))
	parser.add_argument("--pair_chunk_size", type=int, default=int(run_defaults.get("pair_chunk_size", 120)))
	parser.add_argument("--pair_overlap", type=int, default=int(run_defaults.get("pair_overlap", 20)))
	parser.add_argument("--max_pair_chunks", type=int, default=int(run_defaults.get("max_pair_chunks", -1)))
	parser.add_argument("--fps", type=int, default=int(run_defaults.get("fps", 24)))
	parser.add_argument("--near", type=float, default=float(run_defaults.get("near", 0.1)))
	parser.add_argument("--far", type=float, default=float(run_defaults.get("far", 100.0)))
	parser.add_argument("--fx", type=float, default=float(run_defaults.get("fx", 0.86)))
	parser.add_argument("--fy", type=float, default=float(run_defaults.get("fy", 0.86)))
	parser.add_argument("--cx", type=float, default=float(run_defaults.get("cx", 0.5)))
	parser.add_argument("--cy", type=float, default=float(run_defaults.get("cy", 0.5)))
	parser.add_argument("--pnp_opacity_threshold", type=float, default=float(run_defaults.get("pnp_opacity_threshold", 0.3)))
	parser.add_argument("--fallback_baseline", type=float, default=float(run_defaults.get("fallback_baseline", 0.1)))
	parser.add_argument("--save_ply", type=int, default=int(run_defaults.get("save_ply", 1)), help="1 to save NVS reconstruction PLY data")
	parser.add_argument("--save_pair_ply", type=int, default=int(run_defaults.get("save_pair_ply", 0)), help="1 to save one PLY per pair")
	parser.add_argument("--save_merged_ply", type=int, default=int(run_defaults.get("save_merged_ply", 1)), help="1 to save merged PLY")
	parser.add_argument(
		"--ply_use_global_pose_alignment",
		type=int,
		default=int(run_defaults.get("ply_use_global_pose_alignment", 1)),
		help="1 to transform pair points into accumulated global pose before merge",
	)
	parser.add_argument("--ply_opacity_threshold", type=float, default=float(run_defaults.get("ply_opacity_threshold", 0.2)))
	parser.add_argument("--max_points_per_pair", type=int, default=int(run_defaults.get("max_points_per_pair", 4000)))
	parser.add_argument("--max_total_ply_points", type=int, default=int(run_defaults.get("max_total_ply_points", 3000000)))
	return parser


if __name__ == "__main__":
	pre_parser = argparse.ArgumentParser(add_help=False)
	pre_parser.add_argument(
		"--run_args_yaml",
		type=str,
		default=str(_REPO_ROOT / "configs" / "run_urbanscene_nvs_inference.yaml"),
	)
	pre_args, _ = pre_parser.parse_known_args()
	defaults = _load_run_arg_defaults(_resolve_to_repo_path(pre_args.run_args_yaml))

	parser = _build_parser(defaults)
	args = parser.parse_args()
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
	logger.info("[INFO] Startup defaults yaml: %s", args.run_args_yaml)

	runner = ChunkedUrbanSceneNVS(args)
	runner.run()
