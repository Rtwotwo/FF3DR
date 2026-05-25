import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
import cv2


def _find_repo_root(start: Path) -> Path:
    """Find project root by required top-level folders."""
    for p in [start, *start.parents]:
        if (p / "models").is_dir() and (p / "loop_utils").is_dir() and (p / "configs").is_dir():
            return p
    return start.parent


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _find_repo_root(_SCRIPT_DIR)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _resolve_to_repo_path(path_value: str) -> str:
    if path_value is None:
        return path_value
    p = Path(path_value)
    if p.is_absolute():
        return str(p)
    return str((_REPO_ROOT / p).resolve())


from running.utils.config_utils import load_config
from models.vggt.utils.load_fn import load_and_preprocess_images_square
from models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from models.pi3.utils.basic import load_images_as_tensor_pi_long
from models.mapanything.utils.image import load_images

from loop_utils.alignment_torch import apply_sim3_direct_torch, depth_to_point_cloud_optimized_torch
from loop_utils.sim3utils import (
    accumulate_sim3_transforms,
    compute_alignment_error,
    compute_sim3_ab,
    merge_ply_files,
    precompute_scale_chunks_with_depth,
    process_loop_list,
    save_confident_pointcloud_batch,
    weighted_align_point_maps,
)
from loop_utils.loop_detector import LoopDetector

from models.depthanything3.api import DepthAnything3
from models.mapanything.models.mapanything.model import MapAnything
from uniception.models.utils.transformer_blocks import Mlp
from models.pi3.models.pi3 import Pi3
from models.vggt.models.vggt import VGGT
logger = logging.getLogger(__name__)

# unified viz utilities
from running.utils.viz_utils import depth_to_color as _depth_to_color, conf_to_color as _conf_to_color, overlay_depth_on_rgb as _overlay_depth_on_rgb





class FF3DR:
    """Streaming reconstruction for synchronized multi-camera rigs.
    Design goals:
    1) keep frame-major multi-camera ordering;
    2) run robust chunk alignment across time;
    3) support multiple back-end models with one unified output format.
    """
    def __init__(self, args):
        self.args = args
        args.config_path = _resolve_to_repo_path(args.config_path)
        args.area_path = _resolve_to_repo_path(args.area_path)
        args.output_path = _resolve_to_repo_path(args.output_path)
        self.config = load_config(args.config_path)
        self.area_path = args.area_path
        self.model_name = args.model_name
        self.output_path = args.output_path
        if args.device in ["cuda", "cpu"]:
            self.device = args.device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg_chunk_size = int(self.config["Model"]["chunk_size"])
        cfg_overlap = int(self.config["Model"]["overlap"])
        self.chunk_size = args.chunk_size if args.chunk_size > 0 else cfg_chunk_size
        # Use config overlap directly when CLI does not override it.
        self.overlap = args.overlap if args.overlap >= 0 else cfg_overlap
        if self.overlap >= self.chunk_size:
            raise ValueError(f"[SETTING ERROR] overlap={self.overlap} must be smaller than chunk_size={self.chunk_size}")
        if torch.cuda.is_available():
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32
        self.predictions_pcd_dir = os.path.join(self.output_path, "tmp_predictions_pcd")
        self.predictions_loop_dir = os.path.join(self.output_path, "tmp_predictions_loop")
        self.predictions_aligned_dir = os.path.join(self.output_path, "tmp_predictions_aligned")
        self.predictions_unaligned_dir = os.path.join(self.output_path, "tmp_predictions_unaligned")
        self.enable_viz = bool(getattr(args, "enable_viz", False))
        self.viz_max_frames = int(getattr(args, "viz_max_frames", -1))
        self.viz_depth_vmin = getattr(args, "viz_depth_vmin", None)
        self.viz_depth_vmax = getattr(args, "viz_depth_vmax", None)
        self.lora_checkpoint = getattr(args, "lora_checkpoint", None)
        self.process_res = int(getattr(args, "process_res", 518))
        self.process_res_method = str(getattr(args, "process_res_method", "square"))
        self.viz_dir = os.path.join(self.output_path, "viz")
        for p in [self.output_path, self.predictions_pcd_dir, self.predictions_loop_dir, self.predictions_aligned_dir, self.predictions_unaligned_dir]:
            os.makedirs(p, exist_ok=True)
        if self.enable_viz:
            os.makedirs(self.viz_dir, exist_ok=True)
            logger.info("[INFO] Visualization enabled, saving to %s", self.viz_dir)
        self.loop_enable = bool(self.config.get("Model", {}).get("loop_enable", False))
        self.loop_chunk_half_window = int(self.config.get("Model", {}).get("loop_chunk_size", 20))
        self.camera_dirs, self.image_paths = self._build_camera_index(self.area_path)

        # Explicit camera selection has higher priority than num_cameras_to_use.
        requested_camera_ids = [str(x) for x in getattr(args, "camera_ids", []) if str(x) != ""]
        if len(requested_camera_ids) > 0:
            available = {os.path.basename(p): p for p in self.camera_dirs}
            missing = [cid for cid in requested_camera_ids if cid not in available]
            if len(missing) > 0:
                raise ValueError(
                    f"[SETTING ERROR] camera_ids contains missing cameras: {missing}, available={sorted(list(available.keys()), key=lambda x: int(x) if x.isdigit() else x)}"
                )
            self.camera_dirs = [available[cid] for cid in requested_camera_ids]
            self.image_paths = {k: self.image_paths[k] for k in self.camera_dirs}

        self.num_cameras_to_use = args.num_cameras_to_use
        if self.num_cameras_to_use > 0 and len(requested_camera_ids) == 0:
            if self.num_cameras_to_use > len(self.camera_dirs):
                raise ValueError(
                    f"[SETTING ERROR] num_cameras_to_use={self.num_cameras_to_use} exceeds available cameras={len(self.camera_dirs)}"
                )
            self.camera_dirs = self.camera_dirs[: self.num_cameras_to_use]
            self.image_paths = {k: self.image_paths[k] for k in self.camera_dirs}
        self.num_cameras = len(self.camera_dirs)
        self.num_frames = min(len(v) for v in self.image_paths.values())
        self.max_chunks = args.max_chunks
        self.anchor_cam_index = args.anchor_cam_index
        total_images = self.num_cameras * self.num_frames
        selected_camera_names = [os.path.basename(p) for p in self.camera_dirs]
        logger.info(
            "[INFO] Cameras=%d, synced_frames=%d, total_images=%d, chunk_size=%d, overlap=%d, camera_ids=%s",
            self.num_cameras,
            self.num_frames,
            total_images,
            self.chunk_size,
            self.overlap,
            selected_camera_names,
        )
        self.model = self._load_model()
        self.anchor_cam_index = self._resolve_anchor_cam_index(self.anchor_cam_index)
        logger.info("[INFO] Anchor camera index for chunk alignment: %d", self.anchor_cam_index)
        self.da3_infer_mode = args.da3_infer_mode
        self.da3_stable_fusion = bool(args.da3_stable_fusion)
        self.da3_group_frames = max(1, int(args.da3_group_frames))
        self.da3_group_overlap = max(0, int(args.da3_group_overlap))
        if self.da3_group_overlap >= self.da3_group_frames:
            raise ValueError(
                f"[SETTING ERROR] da3_group_overlap={self.da3_group_overlap} must be smaller than da3_group_frames={self.da3_group_frames}"
            )
        if self.da3_infer_mode not in ["anchor_stream", "framewise_multicam", "global_chunk"]:
            raise ValueError(
                f"[SETTING ERROR] da3_infer_mode={self.da3_infer_mode} must be one of anchor_stream/framewise_multicam/global_chunk"
            )
        # Keep compatibility with existing code paths that use this boolean.
        self.da3_use_anchor_stream = (
            self.model_name == "depthanything3"
            and self.num_cameras > 1
            and self.da3_infer_mode == "anchor_stream"
        )
        if self.model_name == "depthanything3" and self.num_cameras > 1:
            logger.info("[INFO] DA3 inference mode: %s", self.da3_infer_mode)
            logger.info("[INFO] DA3 stable fusion: %s", self.da3_stable_fusion)
            logger.info(
                "[INFO] DA3 grouped fusion: group_frames=%d, group_overlap=%d",
                self.da3_group_frames,
                self.da3_group_overlap,
            )
            if self.da3_use_anchor_stream:
                logger.info(
                    "[INFO] DA3 multi-camera mode: using anchor camera stream only (cam=%d) for temporal consistency",
                    self.anchor_cam_index,
                )

    def _cfg_get_weight(self, model_key, field, fallback_flat_key=None):
        """Get weight path from either nested or DA3-style flat config.

        Supported examples:
        - nested: Weights.depthanything3.DA3
        - flat:   Weights.DA3
        """
        weights = self.config.get("Weights", {})
        if model_key in weights and isinstance(weights[model_key], dict) and field in weights[model_key]:
            return weights[model_key][field]
        if fallback_flat_key is not None and fallback_flat_key in weights:
            return weights[fallback_flat_key]
        raise KeyError(f"Missing weight config for {model_key}.{field} (fallback={fallback_flat_key})")

    def _camera_sort_key(self, path):
        name = os.path.basename(path)
        return (0, int(name)) if name.isdigit() else (1, name)

    def _build_camera_index(self, area_path):
        if not os.path.isdir(area_path):
            raise RuntimeError(f"[ERROR] area_path not found: {area_path}")
        camera_dirs = [os.path.join(area_path, d) for d in os.listdir(area_path) if os.path.isdir(os.path.join(area_path, d))]
        camera_dirs = sorted(camera_dirs, key=self._camera_sort_key)
        if len(camera_dirs) == 0:
            raise RuntimeError(f"[ERROR] No camera folders under: {area_path}")
        image_paths = {}
        for cam_dir in camera_dirs:
            files = [
                os.path.join(cam_dir, f)
                for f in os.listdir(cam_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            files = sorted(files)
            if len(files) == 0:
                logger.warning("[WARN] Empty camera folder: %s", cam_dir)
            image_paths[cam_dir] = files
        return camera_dirs, image_paths

    def _load_model(self):
        if self.model_name == "depthanything3":
            da3_config_path = self._cfg_get_weight("depthanything3", "DA3_CONFIG", fallback_flat_key="DA3_CONFIG")
            da3_weight_path = self._cfg_get_weight("depthanything3", "DA3", fallback_flat_key="DA3")
            with open(da3_config_path, "r") as f:
                da3_config = json.load(f)
            model = DepthAnything3(**da3_config)
            state_dict = load_file(da3_weight_path)
            model.load_state_dict(state_dict, strict=False)
            if getattr(self, "lora_checkpoint", None) is not None:
                model = self._apply_lora_to_da3(model)
        elif self.model_name == "mapanything":
            try:
                with open(self.config["Weights"]["mapanything"]["MAP_CONFIG"], "r") as f:
                    map_config = json.load(f)
                # Compat fix: newer uniception expects callable mlp_layer instead of string.
                if isinstance(map_config, dict):
                    info_cfg = map_config.get("info_sharing_config", {})
                    module_args = info_cfg.get("module_args", {})
                    if module_args.get("mlp_layer", None) == "mlp":
                        module_args["mlp_layer"] = Mlp
                model = MapAnything(**map_config)
                state_dict = load_file(self.config["Weights"]["mapanything"]["MAP"])
                model.load_state_dict(state_dict, strict=False)
            except Exception:
                model = MapAnything.from_pretrained(self.config["Weights"]["mapanything"]["MAP_URL"])
        elif self.model_name == "pi3":
            _ = self.config["Weights"]["pi3"]["PI3_CONFIG"]
            model = Pi3()
            state_dict = load_file(self.config["Weights"]["pi3"]["PI3"])
            model.load_state_dict(state_dict, strict=False)
        elif self.model_name == "vggt":
            model = VGGT()
            state_dict = torch.load(self.config["Weights"]["vggt"]["VGGT"], map_location=self.device)
            if isinstance(state_dict, dict):
                model.load_state_dict(state_dict, strict=False)
        else:
            raise RuntimeError("[ERROR] model_name must be one of depthanything3/mapanything/pi3/vggt")
        model.eval().to(self.device)
        logger.info("[INFO] Model loaded: %s", self.model_name)
        return model

    def _apply_lora_to_da3(self, model):
        from peft import LoraConfig, get_peft_model
        try:
            from da3.model.metric_adapter import MetricAdapterV3
        except Exception:
            MetricAdapterV3 = None

        ckpt = torch.load(self.lora_checkpoint, map_location="cpu", weights_only=False)
        ckpt_args = ckpt.get("args", {})
        lora_rank = ckpt_args.get("lora_rank", 16)
        lora_alpha = ckpt_args.get("lora_alpha", 32)
        lora_dropout = ckpt_args.get("lora_dropout", 0.05)
        lora_target_modules = ckpt_args.get("lora_target_modules", ["qkv", "proj"])
        adapter_hidden_dim = ckpt_args.get("adapter_hidden_dim", 64)
        adapter_depth_norm = ckpt_args.get("adapter_depth_norm", 600.0)

        logger.info("[INFO] Applying LoRA: rank=%d alpha=%d dropout=%.3f targets=%s",
                    lora_rank, lora_alpha, lora_dropout, lora_target_modules)

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model.model = get_peft_model(model.model, lora_config)

        if MetricAdapterV3 is not None and hasattr(model.model.base_model.model, "head"):
            head = model.model.base_model.model.head
            if hasattr(head, "metric_adapter"):
                feat_dim = head.metric_adapter.residual_conv1.in_channels - 1
                logger.info("[INFO] Replacing MetricAdapter: feat_dim=%d hidden_dim=%d depth_norm=%.1f",
                             feat_dim, adapter_hidden_dim, adapter_depth_norm)
                head.metric_adapter = MetricAdapterV3(
                    feat_dim=feat_dim,
                    hidden_dim=adapter_hidden_dim,
                    depth_norm=adapter_depth_norm,
                )

        model_sd = ckpt["model_state_dict"]
        da3_sd = {}
        for k, v in model_sd.items():
            if k.startswith("da3."):
                da3_sd[k[len("da3."):]] = v
            elif k.startswith("scale_shift."):
                pass
            else:
                da3_sd[k] = v

        missing, unexpected = model.load_state_dict(da3_sd, strict=False)
        if missing:
            logger.warning("[WARN] Missing keys when loading LoRA checkpoint (%d): %s", len(missing), missing[:5])
        if unexpected:
            logger.warning("[WARN] Unexpected keys when loading LoRA checkpoint (%d): %s", len(unexpected), unexpected[:5])

        logger.info("[INFO] LoRA checkpoint applied: %s", self.lora_checkpoint)
        return model

    def _ensure_batch(self, arr, kind):
        if arr is None:
            return None
        arr = np.asarray(arr)
        if kind in ["depth", "conf"] and arr.ndim == 2:
            return arr[None, ...]
        if kind == "image_coords":
            if arr.ndim == 1:
                return arr[None, ...]
            if arr.ndim == 2 and arr.shape[0] != 1:
                return arr
            if arr.ndim == 3 and arr.shape[0] == 1:
                return arr[0]
            return arr
        if kind == "images" and arr.ndim == 3 and arr.shape[-1] == 3:
            return arr[None, ...]
        if kind == "intrinsics" and arr.ndim == 2:
            return arr[None, ...]
        if kind == "extrinsics" and arr.ndim == 2:
            return arr[None, ...]
        if kind == "world_points" and arr.ndim == 3 and arr.shape[-1] == 3:
            return arr[None, ...]
        return arr

    def _to_numpy(self, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _squeeze_leading_batch(self, value):
        if value is None:
            return None
        value = self._to_numpy(value)
        if value.ndim >= 1 and value.shape[0] == 1:
            return value[0]
        return value

    def _get_prediction_arrays(self, pred):
        if isinstance(pred, dict):
            depth = pred.get("depth", None)
            conf = pred.get("conf", pred.get("depth_conf", pred.get("world_points_conf", None)))
            intrinsics = pred.get("intrinsics", pred.get("intrinsic", None))
            extrinsics = pred.get("extrinsics", pred.get("extrinsic", None))
            images = pred.get("processed_images", pred.get("images", None))
            world_points = pred.get("world_points", None)
            image_coords = pred.get("image_coords", None)
        else:
            depth = getattr(pred, "depth", None)
            conf = getattr(pred, "conf", None)
            intrinsics = getattr(pred, "intrinsics", None)
            extrinsics = getattr(pred, "extrinsics", None)
            images = getattr(pred, "processed_images", getattr(pred, "images", None))
            world_points = getattr(pred, "world_points", None)
            image_coords = getattr(pred, "image_coords", None)

        depth = self._squeeze_leading_batch(depth)
        conf = self._squeeze_leading_batch(conf)
        intrinsics = self._squeeze_leading_batch(intrinsics)
        extrinsics = self._squeeze_leading_batch(extrinsics)
        images = self._squeeze_leading_batch(images)
        world_points = self._squeeze_leading_batch(world_points)

        if depth is not None and depth.ndim == 4 and depth.shape[-1] == 1:
            depth = np.squeeze(depth, axis=-1)
        if conf is not None and conf.ndim == 4 and conf.shape[-1] == 1:
            conf = np.squeeze(conf, axis=-1)
        if extrinsics is not None:
            if extrinsics.ndim == 2 and extrinsics.shape == (4, 4):
                extrinsics = extrinsics[:3, :4]
            elif extrinsics.ndim == 3 and extrinsics.shape[-2:] == (4, 4):
                extrinsics = extrinsics[:, :3, :4]

        depth = self._ensure_batch(depth, "depth")
        conf = self._ensure_batch(conf, "conf")
        intrinsics = self._ensure_batch(intrinsics, "intrinsics")
        extrinsics = self._ensure_batch(extrinsics, "extrinsics")
        images = self._ensure_batch(images, "images")
        world_points = self._ensure_batch(world_points, "world_points")
        image_coords = self._ensure_batch(image_coords, "image_coords") if image_coords is not None else None
        return depth, conf, intrinsics, extrinsics, images, world_points, image_coords

    def _build_chunk_images(self, range_1, range_2=None):
        # Keep frame-major order to ensure one frame includes all cameras.
        frame_indices = list(range(range_1[0], range_1[1]))
        if range_2 is not None:
            frame_indices.extend(list(range(range_2[0], range_2[1])))
        chunk_images = []
        for frame_idx in frame_indices:
            for cam_dir in self.camera_dirs:
                chunk_images.append(self.image_paths[cam_dir][frame_idx])
        return chunk_images

    def _resolve_anchor_cam_index(self, requested_idx):
        """Resolve anchor camera index.

        If user passes a valid index, use it. Otherwise prefer camera folder named '3'
        (typical nadir/center camera in five-view rigs), then fallback to middle index.
        """
        if 0 <= requested_idx < self.num_cameras:
            return requested_idx
        cam_names = [os.path.basename(p) for p in self.camera_dirs]
        if "3" in cam_names:
            return cam_names.index("3")
        return self.num_cameras // 2

    def _select_anchor_from_frame_major(self, arr, overlap_frames, from_tail):
        """Select only anchor-camera entries from frame-major [frame0 cam0..camN-1, ...]."""
        if arr is None:
            return None
        if arr.shape[0] % self.num_cameras != 0:
            # Fallback chunks may provide one sample per frame (anchor stream).
            total_frames = arr.shape[0]
            if overlap_frames <= 0:
                return arr[:0]
            use_frames = min(overlap_frames, total_frames)
            return arr[-use_frames:] if from_tail else arr[:use_frames]
        if self.da3_use_anchor_stream:
            total_frames = arr.shape[0]
            if overlap_frames <= 0:
                return arr[:0]
            use_frames = min(overlap_frames, total_frames)
            return arr[-use_frames:] if from_tail else arr[:use_frames]
        total = arr.shape[0]
        total_frames = total // self.num_cameras
        if overlap_frames <= 0:
            return arr[:0]
        use_frames = min(overlap_frames, total_frames)
        start_frame = total_frames - use_frames if from_tail else 0
        selected = []
        for f in range(start_frame, start_frame + use_frames):
            idx = f * self.num_cameras + self.anchor_cam_index
            selected.append(arr[idx])
        return np.stack(selected, axis=0) if selected else arr[:0]

    def _infer_depthanything3_framewise(self, chunk_images, ref_view_strategy):
        """Run DA3 per frame-group to enforce same-frame multi-view consistency.

        Input chunk_images is frame-major; each frame contributes num_cameras images.
        """
        outputs = {
            "depth": [],
            "conf": [],
            "intrinsics": [],
            "extrinsics": [],
            "processed_images": [],
            "world_points": [],
        }
        total_frames = len(chunk_images) // self.num_cameras
        cum_s = 1.0
        cum_R = np.eye(3, dtype=np.float32)
        cum_t = np.zeros(3, dtype=np.float32)
        prev_cam_centers = None
        frame_align_err_hist = []
        accepted_transforms = 0

        def cam_centers_from_w2c(extrinsics):
            # extrinsics: [N, 3, 4] (w2c)
            R = extrinsics[:, :3, :3]
            t = extrinsics[:, :3, 3]
            Rt = np.transpose(R, (0, 2, 1))
            c = -np.einsum("nij,nj->ni", Rt, t)
            return c.astype(np.float32)

        def estimate_se3_from_points(src, dst):
            # Solve dst ~= R * src + t (after per-frame scale normalization)
            mu_src = src.mean(axis=0)
            mu_dst = dst.mean(axis=0)
            X = src - mu_src
            Y = dst - mu_dst
            cov = (Y.T @ X) / max(src.shape[0], 1)
            U, D, Vt = np.linalg.svd(cov)
            S = np.eye(3, dtype=np.float32)
            if np.linalg.det(U @ Vt) < 0:
                S[2, 2] = -1.0
            R = (U @ S @ Vt).astype(np.float32)
            t = (mu_dst - (R @ mu_src)).astype(np.float32)
            pred = (R @ src.T).T + t[None, :]
            rmse = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))
            return R, t, rmse

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for frame_idx in range(total_frames):
            s = frame_idx * self.num_cameras
            e = s + self.num_cameras
            frame_images = chunk_images[s:e]
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.amp.autocast("cuda", dtype=self.dtype):
                        pred = self.model.inference(frame_images, ref_view_strategy=ref_view_strategy)
                else:
                    pred = self.model.inference(frame_images, ref_view_strategy=ref_view_strategy)

            frame_depth = self._to_numpy(getattr(pred, "depth", None))
            frame_conf = self._to_numpy(getattr(pred, "conf", None)) - 1.0
            frame_intrinsics = self._to_numpy(getattr(pred, "intrinsics", None))
            frame_extrinsics = self._to_numpy(getattr(pred, "extrinsics", None))
            frame_images_np = self._to_numpy(getattr(pred, "processed_images", None))

            frame_points = depth_to_point_cloud_optimized_torch(
                frame_depth,
                frame_intrinsics,
                frame_extrinsics,
            )

            frame_cam_centers = cam_centers_from_w2c(frame_extrinsics)
            if self.num_cameras >= 2:
                rig_scale = float(np.linalg.norm(frame_cam_centers[1] - frame_cam_centers[0]))
            else:
                rig_scale = 1.0
            if not np.isfinite(rig_scale) or rig_scale < 1e-6:
                rig_scale = 1.0
            # Normalize per-frame scale before temporal stitching.
            frame_cam_centers = frame_cam_centers / rig_scale
            frame_points = frame_points / rig_scale

            # Align each frame's multi-camera prediction to the first frame using anchor camera.
            if frame_idx > 0:
                try:
                    R_ij, t_ij, align_err = estimate_se3_from_points(
                        frame_cam_centers,
                        prev_cam_centers,
                    )

                    if len(frame_align_err_hist) < 3:
                        is_bad = align_err > 0.25
                    else:
                        err_thr = max(0.2, 3.0 * float(np.median(frame_align_err_hist)))
                        is_bad = align_err > err_thr

                    if not is_bad:
                        cum_R = R_ij @ cum_R
                        cum_t = (R_ij @ cum_t) + t_ij
                        accepted_transforms += 1
                    else:
                        logger.warning(
                            "[WARN] Framewise stitch outlier at local frame %d: rmse=%.4f, keep previous cumulative transform",
                            frame_idx,
                            align_err,
                        )
                    frame_align_err_hist.append(align_err)
                except Exception as ex:
                    logger.warning(
                        "[WARN] Framewise stitch failed at local frame %d: %s, keep previous cumulative transform",
                        frame_idx,
                        str(ex),
                    )

            frame_points = apply_sim3_direct_torch(frame_points, cum_s, cum_R, cum_t)

            prev_cam_centers = (cum_R @ frame_cam_centers.T).T + cum_t[None, :]

            outputs["depth"].append(frame_depth)
            outputs["conf"].append(frame_conf)
            outputs["intrinsics"].append(frame_intrinsics)
            outputs["extrinsics"].append(frame_extrinsics)
            outputs["processed_images"].append(frame_images_np)
            outputs["world_points"].append(frame_points)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # If temporal stitching is mostly unreliable, fallback to global chunk inference.
        expected = max(total_frames - 1, 1)
        if total_frames > 1 and accepted_transforms < max(3, int(0.15 * expected)):
            logger.warning(
                "[WARN] Framewise stitch unstable (%d/%d accepted). Fallback to anchor stream inference.",
                accepted_transforms,
                expected,
            )
            anchor_images = chunk_images[self.anchor_cam_index :: self.num_cameras]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.amp.autocast("cuda", dtype=self.dtype):
                        pred = self.model.inference(anchor_images, ref_view_strategy=ref_view_strategy)
                else:
                    pred = self.model.inference(anchor_images, ref_view_strategy=ref_view_strategy)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                "depth": self._to_numpy(getattr(pred, "depth", None)),
                "conf": self._to_numpy(getattr(pred, "conf", None)) - 1.0,
                "intrinsics": self._to_numpy(getattr(pred, "intrinsics", None)),
                "extrinsics": self._to_numpy(getattr(pred, "extrinsics", None)),
                "processed_images": self._to_numpy(getattr(pred, "processed_images", None)),
                "world_points": None,
            }

        return {
            "depth": np.concatenate(outputs["depth"], axis=0),
            "conf": np.concatenate(outputs["conf"], axis=0),
            "intrinsics": np.concatenate(outputs["intrinsics"], axis=0),
            "extrinsics": np.concatenate(outputs["extrinsics"], axis=0),
            "processed_images": np.concatenate(outputs["processed_images"], axis=0),
            "world_points": np.concatenate(outputs["world_points"], axis=0),
        }

    def get_chunk_indices(self):
        if self.num_frames <= self.chunk_size:
            return [(0, self.num_frames)]
        step = self.chunk_size - self.overlap
        chunk_indices = []
        for s in range(0, self.num_frames, step):
            e = min(s + self.chunk_size, self.num_frames)
            chunk_indices.append((s, e))
            if e == self.num_frames:
                break
        return chunk_indices

    def _infer_depthanything3(self, chunk_images, ref_view_strategy):
        if self.da3_use_anchor_stream:
            # Input chunk_images is frame-major; pick the same anchor camera for each frame.
            chunk_images = chunk_images[self.anchor_cam_index :: self.num_cameras]
        elif self.model_name == "depthanything3" and self.num_cameras > 1 and self.da3_infer_mode == "framewise_multicam":
            if self.da3_stable_fusion:
                return self._infer_depthanything3_grouped_multicam(chunk_images, ref_view_strategy)
            else:
                return self._infer_depthanything3_framewise(chunk_images, ref_view_strategy)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.amp.autocast("cuda", dtype=self.dtype):
                    pred = self.model.inference(
                        chunk_images,
                        ref_view_strategy=ref_view_strategy,
                        process_res=self.process_res,
                        process_res_method=self.process_res_method,
                    )
            else:
                pred = self.model.inference(
                    chunk_images,
                    ref_view_strategy=ref_view_strategy,
                    process_res=self.process_res,
                    process_res_method=self.process_res_method,
                )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "depth": self._to_numpy(getattr(pred, "depth", None)),
            "conf": self._to_numpy(getattr(pred, "conf", None)) - 1.0,
            "intrinsics": self._to_numpy(getattr(pred, "intrinsics", None)),
            "extrinsics": self._to_numpy(getattr(pred, "extrinsics", None)),
            "processed_images": self._to_numpy(getattr(pred, "processed_images", None)),
            "world_points": None,
        }

    def _infer_depthanything3_grouped_multicam(self, chunk_images, ref_view_strategy):
        """Stable multi-camera fusion with group-wise joint inference + inter-group anchor constraints."""
        total_frames = len(chunk_images) // self.num_cameras
        group_frames = min(self.da3_group_frames, total_frames)
        overlap = min(self.da3_group_overlap, max(group_frames - 1, 0))
        step = max(group_frames - overlap, 1)

        group_ranges = []
        s = 0
        while s < total_frames:
            e = min(s + group_frames, total_frames)
            group_ranges.append((s, e))
            if e == total_frames:
                break
            s += step

        output_lists = {
            "depth": [],
            "conf": [],
            "intrinsics": [],
            "extrinsics": [],
            "processed_images": [],
            "world_points": [],
        }

        prev_group_points = None
        prev_group_conf = None
        cum_s = 1.0
        cum_R = np.eye(3, dtype=np.float32)
        cum_t = np.zeros(3, dtype=np.float32)
        align_err_hist = []

        for gi, (fs, fe) in enumerate(group_ranges):
            group_images = []
            for f in range(fs, fe):
                base = f * self.num_cameras
                group_images.extend(chunk_images[base : base + self.num_cameras])

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.amp.autocast("cuda", dtype=self.dtype):
                        pred = self.model.inference(group_images, ref_view_strategy=ref_view_strategy)
                else:
                    pred = self.model.inference(group_images, ref_view_strategy=ref_view_strategy)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            g_depth = self._to_numpy(getattr(pred, "depth", None))
            g_conf = self._to_numpy(getattr(pred, "conf", None)) - 1.0
            g_intr = self._to_numpy(getattr(pred, "intrinsics", None))
            g_extr = self._to_numpy(getattr(pred, "extrinsics", None))
            g_img = self._to_numpy(getattr(pred, "processed_images", None))
            g_points = depth_to_point_cloud_optimized_torch(g_depth, g_intr, g_extr)

            g_num_frames = fe - fs
            if gi > 0 and overlap > 0 and prev_group_points is not None:
                use_overlap = min(overlap, g_num_frames)
                prev_overlap_points = self._select_anchor_from_frame_major(
                    prev_group_points,
                    use_overlap,
                    from_tail=True,
                )
                prev_overlap_conf = self._select_anchor_from_frame_major(
                    prev_group_conf,
                    use_overlap,
                    from_tail=True,
                )
                cur_overlap_points = self._select_anchor_from_frame_major(
                    g_points,
                    use_overlap,
                    from_tail=False,
                )
                cur_overlap_conf = self._select_anchor_from_frame_major(
                    g_conf,
                    use_overlap,
                    from_tail=False,
                )

                conf_threshold = min(np.median(prev_overlap_conf), np.median(cur_overlap_conf)) * 0.1
                gs, gR, gt = weighted_align_point_maps(
                    prev_overlap_points,
                    prev_overlap_conf,
                    cur_overlap_points,
                    cur_overlap_conf,
                    conf_threshold=conf_threshold,
                    config=self.config,
                    precompute_scale=None,
                )
                gerr = float(
                    compute_alignment_error(
                        prev_overlap_points,
                        prev_overlap_conf,
                        cur_overlap_points,
                        cur_overlap_conf,
                        conf_threshold,
                        gs,
                        gR,
                        gt,
                    )
                )

                if len(align_err_hist) < 2:
                    is_bad = gerr > 0.6
                else:
                    err_thr = max(0.5, 2.8 * float(np.median(align_err_hist)))
                    is_bad = gerr > err_thr

                if not is_bad:
                    cum_s = float(gs * cum_s)
                    cum_R = gR @ cum_R
                    cum_t = gs * (gR @ cum_t) + gt
                else:
                    logger.warning(
                        "[WARN] Group stitch outlier at group %d: err=%.4f, keep previous cumulative transform",
                        gi,
                        gerr,
                    )
                align_err_hist.append(gerr)

            g_points = apply_sim3_direct_torch(g_points, cum_s, cum_R, cum_t)

            # Remove overlap duplicates when appending current group outputs.
            if gi == 0:
                keep_start = 0
            else:
                keep_start = overlap * self.num_cameras

            output_lists["depth"].append(g_depth[keep_start:])
            output_lists["conf"].append(g_conf[keep_start:])
            output_lists["intrinsics"].append(g_intr[keep_start:])
            output_lists["extrinsics"].append(g_extr[keep_start:])
            output_lists["processed_images"].append(g_img[keep_start:])
            output_lists["world_points"].append(g_points[keep_start:])

            prev_group_points = g_points
            prev_group_conf = g_conf

        return {
            "depth": np.concatenate(output_lists["depth"], axis=0),
            "conf": np.concatenate(output_lists["conf"], axis=0),
            "intrinsics": np.concatenate(output_lists["intrinsics"], axis=0),
            "extrinsics": np.concatenate(output_lists["extrinsics"], axis=0),
            "processed_images": np.concatenate(output_lists["processed_images"], axis=0),
            "world_points": np.concatenate(output_lists["world_points"], axis=0),
        }

    def _infer_pi3(self, chunk_images):
        images = load_images_as_tensor_pi_long(
            chunk_images,
            target_size=self.process_res,
            process_res_method=self.process_res_method,
        ).to(self.device)
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.amp.autocast("cuda", dtype=self.dtype):
                    res = self.model(images[None])
            else:
                res = self.model(images[None])
        conf = torch.sigmoid(res["conf"][..., 0])
        depth = res["local_points"][..., 2]
        c2w = res["camera_poses"]
        w2c = torch.linalg.inv(c2w)
        extrinsics = w2c[..., :3, :4]
        b, n, h, w = depth.shape
        intrinsics = torch.zeros((b, n, 3, 3), device=depth.device, dtype=depth.dtype)
        f = float(max(h, w))
        intrinsics[..., 0, 0] = f
        intrinsics[..., 1, 1] = f
        intrinsics[..., 0, 2] = float(w) / 2.0
        intrinsics[..., 1, 2] = float(h) / 2.0
        intrinsics[..., 2, 2] = 1.0
        processed = (images[None].permute(0, 1, 3, 4, 2) * 255.0).clamp(0, 255)
        return {
            "depth": self._to_numpy(depth),
            "conf": self._to_numpy(conf),
            "intrinsics": self._to_numpy(intrinsics),
            "extrinsics": self._to_numpy(extrinsics),
            "processed_images": self._to_numpy(processed.astype(torch.uint8) if hasattr(processed, "astype") else processed.to(torch.uint8)),
            "world_points": self._to_numpy(res["points"]),
        }

    def _infer_vggt(self, chunk_images):
        images_out = load_and_preprocess_images_square(chunk_images, target_size=self.process_res)
        if isinstance(images_out, (tuple, list)):
            images = images_out[0]
            image_coords = images_out[1] if len(images_out) > 1 else None
        else:
            images = images_out
            image_coords = None
        images = images.to(self.device)
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.amp.autocast("cuda", dtype=self.dtype):
                    pred = self.model(images)
            else:
                pred = self.model(images)
        pose_enc = pred["pose_enc"]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        depth = pred["depth"][..., 0]
        conf = pred.get("depth_conf", pred.get("world_points_conf"))
        processed = (images.unsqueeze(0).permute(0, 1, 3, 4, 2) * 255.0).clamp(0, 255)
        return {
            "depth": self._to_numpy(depth),
            "conf": self._to_numpy(conf),
            "intrinsics": self._to_numpy(intrinsics),
            "extrinsics": self._to_numpy(extrinsics),
            "processed_images": self._to_numpy(processed.to(torch.uint8)),
            "world_points": self._to_numpy(pred.get("world_points", None)),
            "image_coords": self._to_numpy(image_coords),
        }

    def _infer_mapanything(self, chunk_images):
        views = load_images(chunk_images, resize_mode="square", size=self.process_res)
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

        def cat_tensors(key):
            return torch.cat([p[key] for p in pred_list], dim=0)

        world_points = cat_tensors("pts3d").unsqueeze(0)
        conf = cat_tensors("conf").unsqueeze(0)
        depth = cat_tensors("depth_z").squeeze(-1).unsqueeze(0)
        intrinsics = cat_tensors("intrinsics").unsqueeze(0)
        c2w = cat_tensors("camera_poses")
        w2c = torch.linalg.inv(c2w)
        extrinsics = w2c[:, :3, :4].unsqueeze(0)
        images = (cat_tensors("img_no_norm") * 255.0).clamp(0, 255).to(torch.uint8).unsqueeze(0)
        return {
            "depth": self._to_numpy(depth),
            "conf": self._to_numpy(conf),
            "intrinsics": self._to_numpy(intrinsics),
            "extrinsics": self._to_numpy(extrinsics),
            "processed_images": self._to_numpy(images),
            "world_points": self._to_numpy(world_points),
        }

    def inference_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        chunk_images = self._build_chunk_images(range_1, range_2)
        ref_view_strategy = self.config["Model"]["ref_view_strategy" if not is_loop else "ref_view_strategy_loop"]
        if self.model_name == "depthanything3":
            predictions = self._infer_depthanything3(chunk_images, ref_view_strategy)
        elif self.model_name == "mapanything":
            predictions = self._infer_mapanything(chunk_images)
        elif self.model_name == "pi3":
            predictions = self._infer_pi3(chunk_images)
        elif self.model_name == "vggt":
            predictions = self._infer_vggt(chunk_images)
        else:
            raise RuntimeError(f"[ERROR] Unsupported model: {self.model_name}")
        if is_loop:
            save_dir = self.predictions_loop_dir
            file_name = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy" if range_2 is not None else f"loop_{range_1[0]}_{range_1[1]}.npy"
        else:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided when is_loop is False")
            save_dir = self.predictions_unaligned_dir
            file_name = f"chunk_{chunk_idx}.npy"
        save_path = os.path.join(save_dir, file_name)
        np.save(save_path, predictions, allow_pickle=True)
        logger.info("[INFO] Saved chunk %s with %d images", file_name, len(chunk_images))
        return predictions

    def align_2pcds(self, point_map1, conf1, point_map2, conf2, chunk1_depth, chunk2_depth, chunk1_depth_conf, chunk2_depth_conf):
        conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1
        scale_factor = None
        if self.config["Model"]["align_method"] == "scale+se3":
            scale_factor, quality_score, method_used = precompute_scale_chunks_with_depth(
                chunk1_depth,
                chunk1_depth_conf,
                chunk2_depth,
                chunk2_depth_conf,
                method=self.config["Model"]["scale_compute_method"],
            )
            logger.info("[INFO] Depth-scale precompute: scale=%s, quality=%s, method=%s", scale_factor, quality_score, method_used)
        s, R, t = weighted_align_point_maps(
            point_map1,
            conf1,
            point_map2,
            conf2,
            conf_threshold=conf_threshold,
            config=self.config,
            precompute_scale=scale_factor,
        )
        mean_error = compute_alignment_error(
            point_map1,
            conf1,
            point_map2,
            conf2,
            conf_threshold,
            s,
            R,
            t,
        )
        return s, R, t, float(mean_error)

    def _rotation_angle_deg(self, R):
        tr = float(np.trace(R))
        cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    def _is_anchor_sim3_outlier(self, s, R, t, mean_error, prev_sim3, err_hist):
        # Reject chunk-pair transforms that are highly likely to be mismatches.
        if prev_sim3 is None:
            return False
        if len(err_hist) < 3:
            # Warmup stage: only reject very obvious failures.
            return mean_error > 0.9

        med_err = float(np.median(err_hist))
        err_threshold = max(0.55, 3.2 * med_err)
        return mean_error > err_threshold

    def _refine_aligned_overlap(self, prev_points, prev_conf, cur_points, cur_conf):
        """Estimate a small seam correction that maps current overlap to previous overlap."""
        conf_threshold = min(np.median(prev_conf), np.median(cur_conf)) * 0.1
        refine_cfg = copy.deepcopy(self.config)
        refine_cfg["Model"]["align_method"] = "se3"

        s, R, t = weighted_align_point_maps(
            prev_points,
            prev_conf,
            cur_points,
            cur_conf,
            conf_threshold=conf_threshold,
            config=refine_cfg,
            precompute_scale=None,
        )
        mean_error = compute_alignment_error(
            prev_points,
            prev_conf,
            cur_points,
            cur_conf,
            conf_threshold,
            s,
            R,
            t,
        )
        return s, R, t, float(mean_error)

    def _chunk_to_point_arrays(self, pred):
        depth, conf, intrinsics, extrinsics, images, world_points, image_coords = self._get_prediction_arrays(pred)
        if conf is None:
            raise RuntimeError("[ERROR] Missing confidence map in prediction")
        if world_points is None:
            if depth is None or intrinsics is None or extrinsics is None:
                raise RuntimeError("[ERROR] Missing depth/intrinsics/extrinsics to build world points")
            world_points = depth_to_point_cloud_optimized_torch(depth, intrinsics, extrinsics)
        if depth is None:
            depth = np.linalg.norm(world_points, axis=-1)
        if images is not None and images.dtype != np.uint8:
            if np.max(images) <= 1.0:
                images = (images * 255.0).clip(0, 255).astype(np.uint8)
            else:
                images = images.clip(0, 255).astype(np.uint8)
        return world_points, conf, images, depth

    @staticmethod
    def _infer_content_bbox(image):
        if image is None:
            return None
        img = np.asarray(image)
        if img.ndim == 2:
            img = img[..., None]
        if img.ndim != 3 or img.shape[0] < 8 or img.shape[1] < 8:
            return None
        if img.dtype != np.uint8:
            img = np.clip(img, 0.0, 255.0 if float(np.nanmax(img)) > 1.5 else 1.0)
            if float(np.nanmax(img)) <= 1.5:
                img = (img * 255.0).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        corner_patch = np.concatenate([
            img[:4, :4].reshape(-1, img.shape[-1]),
            img[:4, -4:].reshape(-1, img.shape[-1]),
            img[-4:, :4].reshape(-1, img.shape[-1]),
            img[-4:, -4:].reshape(-1, img.shape[-1]),
        ], axis=0)
        pad_color = np.median(corner_patch, axis=0)
        dist = np.abs(img.astype(np.int16) - pad_color.astype(np.int16)).sum(axis=-1)
        mask = dist > 18
        if not np.any(mask):
            return None
        ys, xs = np.where(mask)
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _crop_to_bbox(arr, bbox):
        if arr is None or bbox is None:
            return arr
        x1, y1, x2, y2 = bbox
        return arr[y1:y2, x1:x2]

    @staticmethod
    def _resize_viz_image(image, target_size):
        if image is None:
            return None
        arr = np.asarray(image)
        if arr.size == 0:
            return arr
        if arr.shape[0] == target_size and arr.shape[1] == target_size:
            return arr
        return cv2.resize(arr, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

    def _save_viz_for_chunk(self, chunk_data, chunk_idx, frame_offset):
        if not self.enable_viz:
            return
        depth, conf, intrinsics, extrinsics, images, _, image_coords = self._get_prediction_arrays(chunk_data)
        if depth is None or conf is None:
            return
        n_views = depth.shape[0]
        cam_names = [os.path.basename(p) for p in self.camera_dirs]
        viz_target_size = int(getattr(self, "viz_output_size", self.process_res))

        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        if conf.ndim == 4 and conf.shape[-1] == 1:
            conf = conf.squeeze(-1)

        crop_bboxes = []
        for vi in range(n_views):
            bbox = None
            if image_coords is not None:
                coord = np.asarray(image_coords[vi]).reshape(-1)
                if coord.size >= 4 and np.all(np.isfinite(coord[:4])):
                    x1, y1, x2, y2 = coord[:4]
                    bbox = (
                        int(np.floor(x1)),
                        int(np.floor(y1)),
                        int(np.ceil(x2)),
                        int(np.ceil(y2)),
                    )
            if bbox is None and images is not None:
                bbox = self._infer_content_bbox(images[vi])
            crop_bboxes.append(bbox)

        if self.viz_depth_vmin is not None and self.viz_depth_vmax is not None:
            depth_global_min = float(self.viz_depth_vmin)
            depth_global_max = float(self.viz_depth_vmax)
        else:
            valid_depth_list = []
            for vi in range(n_views):
                d = self._crop_to_bbox(depth[vi], crop_bboxes[vi])
                if d is None:
                    continue
                valid = d[np.isfinite(d) & (d > 0)]
                if valid.size > 0:
                    valid_depth_list.append(valid)

            valid_depth = np.concatenate(valid_depth_list, axis=0) if len(valid_depth_list) > 0 else np.array([], dtype=np.float32)
            if valid_depth.size > 0:
                depth_global_min = float(np.quantile(valid_depth, 0.02))
                depth_global_max = float(np.quantile(valid_depth, 0.98))
            else:
                depth_global_min = 0.0
                depth_global_max = 1.0

        for vi in range(n_views):
            if self.num_cameras > 1:
                frame_local = vi // self.num_cameras
                cam_local = vi % self.num_cameras
                cam_name = cam_names[cam_local] if cam_local < len(cam_names) else str(cam_local)
            else:
                frame_local = vi
                cam_name = cam_names[0] if len(cam_names) > 0 else "0"

            global_frame_idx = frame_offset + frame_local
            if self.viz_max_frames > 0 and global_frame_idx >= self.viz_max_frames:
                continue
            stem = "{:06d}".format(global_frame_idx)

            cam_viz_dir = os.path.join(self.viz_dir, cam_name)
            os.makedirs(cam_viz_dir, exist_ok=True)

            bbox = crop_bboxes[vi]
            d = self._crop_to_bbox(depth[vi], bbox)
            c = self._crop_to_bbox(conf[vi], bbox)

            depth_color = _depth_to_color(d, vmin=depth_global_min, vmax=depth_global_max)
            depth_color = self._resize_viz_image(depth_color, viz_target_size)
            cv2.imwrite(os.path.join(cam_viz_dir, "{}_depth.png".format(stem)), depth_color)

            conf_color = _conf_to_color(c)
            conf_color = self._resize_viz_image(conf_color, viz_target_size)
            cv2.imwrite(os.path.join(cam_viz_dir, "{}_conf.png".format(stem)), conf_color)

            if images is not None:
                img = self._crop_to_bbox(images[vi], bbox)
                if img.ndim == 3 and img.shape[-1] == 3:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif img.ndim == 2:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    img_bgr = None
                if img_bgr is not None:
                    img_bgr = self._resize_viz_image(img_bgr, viz_target_size)
                    cv2.imwrite(os.path.join(cam_viz_dir, "{}_rgb.png".format(stem)), img_bgr)
                    overlay = _overlay_depth_on_rgb(img_bgr, depth_color, alpha=0.5)
                    overlay = self._resize_viz_image(overlay, viz_target_size)
                    cv2.imwrite(os.path.join(cam_viz_dir, "{}_overlay.png".format(stem)), overlay)

        logger.info("[INFO] Saved viz for chunk %d (%d views) to %s", chunk_idx, n_views, self.viz_dir)

    def _run_loop_detection(self, chunk_indices):
        """Run SALAD-based loop detection on the image sequence.

        For multi-camera rigs, uses the anchor camera's image directory.
        Returns a list of (chunk_idx1, range1, chunk_idx2, range2) tuples
        representing loop closure pairs with their local frame ranges.
        """
        anchor_dir = self.camera_dirs[self.anchor_cam_index] if self.num_cameras > 1 else self.area_path
        loop_output = os.path.join(self.output_path, "loop_closures.txt")
        detector = LoopDetector(
            image_dir=anchor_dir,
            output=loop_output,
            config=self.config,
        )
        logger.info("[INFO] Running loop detection on: %s", anchor_dir)
        res = detector.run()
        # detector.run() may return early (None) when no images are found.
        if res is None or getattr(detector, "loop_closures", None) is None:
            logger.info("[INFO] No images found for loop detection or no loop closures")
            return []
        loop_list = detector.get_loop_list()
        if len(loop_list) == 0:
            logger.info("[INFO] No loop closures detected")
            return []
        logger.info("[INFO] Detected %d loop closure pairs", len(loop_list))
        processed = process_loop_list(chunk_indices, loop_list, half_window=self.loop_chunk_half_window)
        logger.info("[INFO] Processed %d loop pairs into chunk frame ranges", len(processed))
        return processed

    def _compute_loop_sim3(self, loop_pairs, chunk_indices, cumulative_sim3):
        """Infer loop chunks and compute relative SIM3 transforms for each loop pair.

        For each loop pair (chunk_i, range_i, chunk_j, range_j):
        1. Run inference on both ranges (is_loop=True).
        2. Compute SIM3 between the two loop predictions.
        3. Compare with the accumulated SIM3 from sequential alignment.
        4. Compute a correction transform.

        Returns a list of (chunk_idx_j, correction_sim3) for global optimization.
        """
        loop_sim3_corrections = []
        for pair_idx, (ci, range_i, cj, range_j) in enumerate(loop_pairs):
            logger.info(
                "[Loop] pair %d/%d: chunk_i=%d range=%s, chunk_j=%d range=%s",
                pair_idx + 1, len(loop_pairs), ci, range_i, cj, range_j,
            )
            try:
                pred_i = self.inference_single_chunk(range_i, is_loop=True)
                pred_j = self.inference_single_chunk(range_j, is_loop=True)
            except Exception as e:
                logger.warning("[Loop] Inference failed for pair %d: %s", pair_idx, str(e))
                continue

            pts_i, conf_i, _, depth_i = self._chunk_to_point_arrays(pred_i)
            pts_j, conf_j, _, depth_j = self._chunk_to_point_arrays(pred_j)

            try:
                s_loop, R_loop, t_loop, err_loop = self.align_2pcds(
                    pts_i, conf_i, pts_j, conf_j,
                    np.squeeze(depth_i) if depth_i is not None else None,
                    np.squeeze(depth_j) if depth_j is not None else None,
                    np.squeeze(conf_i), np.squeeze(conf_j),
                )
            except Exception as e:
                logger.warning("[Loop] Alignment failed for pair %d: %s", pair_idx, str(e))
                continue

            logger.info("[Loop] pair %d: alignment error=%.4f, s=%.4f", pair_idx, err_loop, s_loop)

            S_loop = (s_loop, R_loop, t_loop)

            if ci < len(cumulative_sim3) and cj > 0 and (cj - 1) < len(cumulative_sim3):
                S_i = cumulative_sim3[ci] if ci > 0 else (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
                S_j = cumulative_sim3[cj - 1]
                S_ij_sequential = compute_sim3_ab(S_i, S_j)
                S_correction = compute_sim3_ab(S_ij_sequential, S_loop)
                loop_sim3_corrections.append((cj, S_correction, err_loop))
                logger.info(
                    "[Loop] pair %d: correction s=%.4f, rot=%.2fdeg, |t|=%.4f",
                    pair_idx,
                    S_correction[0],
                    self._rotation_angle_deg(S_correction[1]),
                    float(np.linalg.norm(S_correction[2])),
                )
            else:
                logger.warning("[Loop] pair %d: chunk indices out of cumulative_sim3 range, skipping", pair_idx)

        return loop_sim3_corrections

    def _apply_loop_corrections(self, cumulative_sim3, loop_corrections, num_chunks):
        """Apply loop closure corrections to the cumulative SIM3 transforms.

        Uses a weighted averaging approach: for each chunk that has a loop
        correction, blend the correction into the cumulative transform.
        The blending weight is inversely proportional to the loop alignment error.
        """
        if len(loop_corrections) == 0:
            return cumulative_sim3

        correction_map = {}
        for cj, S_corr, err in loop_corrections:
            if cj not in correction_map or err < correction_map[cj][1]:
                correction_map[cj] = (S_corr, err)

        corrected = list(cumulative_sim3)
        for cj, (S_corr, err) in correction_map.items():
            if cj - 1 < len(corrected):
                s_c, R_c, t_c = corrected[cj - 1]
                s_corr, R_corr, t_corr = S_corr
                w = min(1.0, 0.3 / (err + 1e-6))
                w = min(w, 0.5)
                s_new = s_c * (1.0 - w) + (s_c * s_corr) * w
                R_new = R_c
                t_new = t_c * (1.0 - w) + (s_corr * (R_corr @ t_c) + t_corr) * w
                corrected[cj - 1] = (s_new, R_new, t_new)
                logger.info(
                    "[Loop] Corrected chunk %d: w=%.3f, s=%.4f->%.4f",
                    cj, w, s_c, s_new,
                )

        return corrected

    def process_long_sequence(self):
        """Run chunked reconstruction and align chunk coordinate systems with SIM3."""
        if self.num_frames == 0:
            raise RuntimeError("[ERROR] No frames found")
        chunk_indices = self.get_chunk_indices()
        if self.max_chunks > 0:
            chunk_indices = chunk_indices[: self.max_chunks]
        logger.info("[INFO] Processing %d frames in %d chunks", self.num_frames, len(chunk_indices))
        pre_predictions = None
        sim3_list = []
        sim3_error_hist = []
        prev_good_sim3 = None
        for chunk_idx, fr in enumerate(chunk_indices):
            logger.info("[Progress] chunk %d/%d, frame_range=%s", chunk_idx + 1, len(chunk_indices), fr)
            cur_predictions = self.inference_single_chunk(fr, chunk_idx=chunk_idx, is_loop=False)
            if self.enable_viz:
                frame_offset = fr[0]
                self._save_viz_for_chunk(cur_predictions, chunk_idx, frame_offset)
            if pre_predictions is not None:
                points1, conf1, _, depth1 = self._chunk_to_point_arrays(pre_predictions)
                points2, conf2, _, depth2 = self._chunk_to_point_arrays(cur_predictions)
                # Important: use only one anchor camera per frame for temporal SIM3.
                # Mixing all view directions in overlap often makes registration unstable.
                overlap_frames = self.overlap
                if overlap_frames > 0:
                    point_map1 = self._select_anchor_from_frame_major(points1, overlap_frames, from_tail=True)
                    point_map2 = self._select_anchor_from_frame_major(points2, overlap_frames, from_tail=False)
                    conf_map1 = self._select_anchor_from_frame_major(conf1, overlap_frames, from_tail=True)
                    conf_map2 = self._select_anchor_from_frame_major(conf2, overlap_frames, from_tail=False)
                    if self.config["Model"]["align_method"] == "scale+se3":
                        depth_map1 = np.squeeze(self._select_anchor_from_frame_major(depth1, overlap_frames, from_tail=True))
                        depth_map2 = np.squeeze(self._select_anchor_from_frame_major(depth2, overlap_frames, from_tail=False))
                        depth_conf1 = np.squeeze(conf_map1)
                        depth_conf2 = np.squeeze(conf_map2)
                    else:
                        depth_map1 = None
                        depth_map2 = None
                        depth_conf1 = None
                        depth_conf2 = None
                    s, R, t, mean_error = self.align_2pcds(
                        point_map1,
                        conf_map1,
                        point_map2,
                        conf_map2,
                        depth_map1,
                        depth_map2,
                        depth_conf1,
                        depth_conf2,
                    )
                    if self.da3_use_anchor_stream:
                        if self._is_anchor_sim3_outlier(
                            s,
                            R,
                            t,
                            mean_error,
                            prev_good_sim3,
                            sim3_error_hist,
                        ):
                            if prev_good_sim3 is not None:
                                logger.warning(
                                    "[WARN] Outlier SIM3 at chunk %d: err=%.4f, fallback to previous stable transform",
                                    chunk_idx,
                                    mean_error,
                                )
                                s, R, t = prev_good_sim3
                        else:
                            prev_good_sim3 = (s, R, t)
                        sim3_error_hist.append(mean_error)
                    sim3_list.append((s, R, t))
                else:
                    sim3_list.append((1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)))
            pre_predictions = cur_predictions
        if len(sim3_list) > 0:
            cumulative_sim3 = accumulate_sim3_transforms(sim3_list)
        else:
            cumulative_sim3 = []

        # ── Loop closure detection and correction ──────────────────────────
        if self.loop_enable and len(chunk_indices) > 1:
            logger.info("[INFO] Loop closure detection enabled")
            loop_pairs = self._run_loop_detection(chunk_indices)
            if len(loop_pairs) > 0:
                loop_corrections = self._compute_loop_sim3(loop_pairs, chunk_indices, cumulative_sim3)
                if len(loop_corrections) > 0:
                    cumulative_sim3 = self._apply_loop_corrections(cumulative_sim3, loop_corrections, len(chunk_indices))
                    logger.info("[INFO] Applied %d loop corrections to cumulative SIM3", len(loop_corrections))
                else:
                    logger.info("[INFO] No valid loop corrections computed")
            else:
                logger.info("[INFO] No loop pairs to process")

        prev_aligned_world_points = None
        prev_aligned_confs = None
        for chunk_idx in range(len(chunk_indices)):
            chunk_data = np.load(os.path.join(self.predictions_unaligned_dir, f"chunk_{chunk_idx}.npy"), allow_pickle=True).item()
            world_points, confs, colors, _ = self._chunk_to_point_arrays(chunk_data)
            if chunk_idx > 0:
                s, R, t = cumulative_sim3[chunk_idx - 1]
                world_points = apply_sim3_direct_torch(world_points, s, R, t)
                if self.da3_use_anchor_stream and self.overlap > 0 and prev_aligned_world_points is not None:
                    overlap_n = min(self.overlap, prev_aligned_world_points.shape[0], world_points.shape[0])
                    if overlap_n > 0:
                        prev_overlap_pts = prev_aligned_world_points[-overlap_n:]
                        prev_overlap_conf = prev_aligned_confs[-overlap_n:]
                        cur_overlap_pts = world_points[:overlap_n]
                        cur_overlap_conf = confs[:overlap_n]
                        try:
                            rs, rR, rt, rerr = self._refine_aligned_overlap(
                                prev_overlap_pts,
                                prev_overlap_conf,
                                cur_overlap_pts,
                                cur_overlap_conf,
                            )
                            rot_deg = self._rotation_angle_deg(rR)
                            # Keep refinement conservative to avoid over-correction.
                            if rerr < 0.45 and abs(rs - 1.0) < 0.03 and rot_deg < 3.0 and np.linalg.norm(rt) < 1.5:
                                world_points = apply_sim3_direct_torch(world_points, rs, rR, rt)
                                logger.info(
                                    "[INFO] Seam refine chunk %d: err=%.4f, rot=%.2fdeg, |t|=%.3f",
                                    chunk_idx,
                                    rerr,
                                    rot_deg,
                                    float(np.linalg.norm(rt)),
                                )
                        except Exception as e:
                            logger.warning("[WARN] Seam refine failed at chunk %d: %s", chunk_idx, str(e))
            aligned_chunk = {"world_points": world_points, "conf": confs, "images": colors}
            np.save(os.path.join(self.predictions_aligned_dir, f"chunk_{chunk_idx}.npy"), aligned_chunk, allow_pickle=True)
            conf_threshold = np.mean(confs) * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"]
            ply_path = os.path.join(self.predictions_pcd_dir, f"{chunk_idx}_pcd.ply")
            save_confident_pointcloud_batch(
                points=world_points,
                colors=colors,
                confs=confs,
                output_path=ply_path,
                conf_threshold=conf_threshold,
                sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
            )
            logger.info("[INFO] Saved aligned PLY: %s", ply_path)
            prev_aligned_world_points = world_points
            prev_aligned_confs = confs
        merged_path = os.path.join(self.output_path, "reconstruction_merged.ply")
        merge_ply_files(self.predictions_pcd_dir, merged_path)
        logger.info("[INFO] Reconstruction done: %s", merged_path)
        if bool(self.config.get("Model", {}).get("delete_temp_files", False)):
            self._cleanup_temp_npy_files()

    def run(self):
        logger.info("[INFO] Running FF3DR on %s", self.area_path)
        self.process_long_sequence()

    def _cleanup_temp_npy_files(self):
        target_dirs = [
            self.predictions_unaligned_dir,
            self.predictions_aligned_dir,
            self.predictions_loop_dir,
        ]
        removed_files = 0
        removed_bytes = 0
        for d in target_dirs:
            if not os.path.isdir(d):
                continue
            for name in os.listdir(d):
                if not name.endswith(".npy"):
                    continue
                fpath = os.path.join(d, name)
                try:
                    removed_bytes += os.path.getsize(fpath)
                    os.remove(fpath)
                    removed_files += 1
                except Exception as e:
                    logger.warning("[WARN] Failed to delete temp file %s: %s", fpath, str(e))
        logger.info(
            "[INFO] Temp cleanup done: removed %d npy files (%.2f MB)",
            removed_files,
            removed_bytes / (1024.0 * 1024.0),
        )


def _load_run_arg_defaults(yaml_path):
    if yaml_path is None or not os.path.isfile(yaml_path):
        return {}
    cfg = load_config(yaml_path)
    if not isinstance(cfg, dict):
        return {}
    # Support both flat and nested style under RunArgs.
    defaults = cfg.get("RunArgs", cfg)
    return defaults if isinstance(defaults, dict) else {}


def _build_parser(run_defaults):
    parser = argparse.ArgumentParser(description="FF3DR")
    parser.add_argument(
        "--run_args_yaml",
        type=str,
        default=str(_REPO_ROOT / "configs" / "run_whuomvs_inference.yaml"),
        help="YAML file with default startup arguments",
    )
    parser.add_argument(
        "--area_path",
        type=str,
        default=run_defaults.get("area_path", "./dataset/WHU-OMVS/train/area1/images"),
        help="Area images folder, containing camera subfolders",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=run_defaults.get("config_path", str(_REPO_ROOT / "configs" / "base_config.yaml")),
        help="Config file path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=run_defaults.get("output_path", "./exp"),
        help="Output directory",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=run_defaults.get("model_name", "depthanything3"),
        help="depthanything3/mapanything/pi3/vggt",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=int(run_defaults.get("chunk_size", -1)),
        help="Chunk size in frames, <=0 to use config",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=int(run_defaults.get("overlap", -1)),
        help="Chunk overlap in frames, -1 to use config",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=int(run_defaults.get("max_chunks", -1)),
        help="Run only first N chunks for quick debug, -1 means all",
    )
    parser.add_argument(
        "--anchor_cam_index",
        type=int,
        default=int(run_defaults.get("anchor_cam_index", -1)),
        help="Anchor camera index for chunk alignment, -1 means auto",
    )
    parser.add_argument(
        "--num_cameras_to_use",
        type=int,
        default=int(run_defaults.get("num_cameras_to_use", -1)),
        help="How many sorted camera folders to use, -1 means all",
    )
    parser.add_argument(
        "--camera_ids",
        type=str,
        nargs="*",
        default=run_defaults.get("camera_ids", []),
        help="Explicit camera folder names to use, e.g. --camera_ids 1 3 5; higher priority than num_cameras_to_use",
    )
    parser.add_argument(
        "--da3_infer_mode",
        type=str,
        default=run_defaults.get("da3_infer_mode", "framewise_multicam"),
        help="DA3 mode: framewise_multicam/anchor_stream/global_chunk",
    )
    parser.add_argument(
        "--da3_stable_fusion",
        type=int,
        default=int(run_defaults.get("da3_stable_fusion", 1)),
        help="1: force stable fusion path for DA3 multicam (recommended), 0: allow raw framewise stitching",
    )
    parser.add_argument(
        "--da3_group_frames",
        type=int,
        default=int(run_defaults.get("da3_group_frames", 8)),
        help="Frames per multicam inference group when stable fusion is enabled",
    )
    parser.add_argument(
        "--da3_group_overlap",
        type=int,
        default=int(run_defaults.get("da3_group_overlap", 2)),
        help="Frame overlap between adjacent multicam inference groups",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=run_defaults.get("device", "cpu"),
        help="Fallback device string when cuda is unavailable",
    )
    parser.add_argument(
        "--enable_viz",
        action="store_true",
        default=bool(run_defaults.get("enable_viz", False)),
        help="Save per-frame visualization (depth/conf/rgb/overlay) under <output>/viz/",
    )
    parser.add_argument(
        "--viz_max_frames",
        type=int,
        default=int(run_defaults.get("viz_max_frames", -1)),
        help="Maximum number of frame indices to save viz for; -1 means no limit.",
    )
    parser.add_argument(
        "--viz_depth_vmin",
        type=float,
        default=run_defaults.get("viz_depth_vmin", None),
        help="Fixed minimum depth for visualization normalization; None uses per-chunk quantiles.",
    )
    parser.add_argument(
        "--viz_depth_vmax",
        type=float,
        default=run_defaults.get("viz_depth_vmax", None),
        help="Fixed maximum depth for visualization normalization; None uses per-chunk quantiles.",
    )
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default=run_defaults.get("lora_checkpoint", None),
        help="Optional LoRA checkpoint to apply to DA3 model for visualization.",
    )
    parser.add_argument(
        "--process_res",
        type=int,
        default=int(run_defaults.get("process_res", 518)),
        help="Shared preprocessing resolution for all backends.",
    )
    parser.add_argument(
        "--process_res_method",
        type=str,
        default=run_defaults.get("process_res_method", "square"),
        help="Shared preprocessing method for all backends.",
    )
    return parser


if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--run_args_yaml",
        type=str,
        default=str(_REPO_ROOT / "configs" / "run_whuomvs_inference.yaml"),
    )
    pre_args, _ = pre_parser.parse_known_args()
    defaults = _load_run_arg_defaults(pre_args.run_args_yaml)

    parser = _build_parser(defaults)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger.info("[INFO] Startup defaults yaml: %s", args.run_args_yaml)
    ff3dr = FF3DR(args)
    ff3dr.run()