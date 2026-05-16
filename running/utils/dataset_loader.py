"""Universal dataset loader for MatrixCity, WHU-OMVS, and UrbanScene.

Supports three dataset formats with auto-detection:

1. MatrixCity  – flat folder of sequential PNG images + transforms.json (NeRF format)
2. WHU-OMVS    – multi-camera sub-directories (images/{1..5}/) + COLMAP-style camera info
3. UrbanScene  – flat folder of JPG/PNG images, no ground-truth

Typical usage::

    from running.utils.dataset_loader import DatasetLoader

    loader = DatasetLoader(
        dataset_path="./dataset/MatrixCity/small_city/aerial/train/block_1",
        image_size=518,
        patch_size=14,
    )
    result = loader.load()
    images = result["images"]          # [N, 3, H, W]  or  [N*C, 3, H, W] (frame-major)
    dataset_type = result["dataset_type"]
    num_cameras = result["num_cameras"]
    num_frames = result["num_frames"]
"""

import glob
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "models").is_dir() and (p / "configs").is_dir():
            return p
    return start.parent


_REPO_ROOT = _find_repo_root(Path(__file__).resolve())


def _import_load_and_preprocess_images():
    for candidate in [
        "models.lingbot_map.utils.load_fn",
        "models.vggt.utils.load_fn",
    ]:
        try:
            mod = __import__(candidate, fromlist=["load_and_preprocess_images"])
            return mod.load_and_preprocess_images
        except (ImportError, AttributeError):
            continue
    raise ImportError(
        "Cannot import load_and_preprocess_images from any known module. "
        "Ensure models/lingbot_map or models/vggt is on sys.path."
    )


_load_and_preprocess_images = None


def _get_load_fn():
    global _load_and_preprocess_images
    if _load_and_preprocess_images is None:
        _load_and_preprocess_images = _import_load_and_preprocess_images()
    return _load_and_preprocess_images


# =============================================================================
# Dataset type detection
# =============================================================================

def detect_dataset_type(dataset_path: str) -> str:
    """Auto-detect dataset type from directory structure.

    Returns one of ``"matrixcity"``, ``"whu_omvs"``, ``"urbanscene"``.

    Detection heuristics:
    - WHU-OMVS: path contains ``WHU-OMVS`` or sub-dirs named like numeric
      camera indices (1, 2, 3, 4, 5) containing images.
    - MatrixCity: path contains ``MatrixCity`` or a ``transforms.json`` exists.
    - UrbanScene: path contains ``UrbanScene`` or falls through as default.
    """
    p = Path(dataset_path).resolve()

    if "WHU-OMVS" in str(p) or "whu_omvs" in str(p).lower():
        return "whu_omvs"

    if "MatrixCity" in str(p) or "matrixcity" in str(p).lower():
        return "matrixcity"

    if "UrbanScene" in str(p) or "urbanscene" in str(p).lower():
        return "urbanscene"

    if (p / "transforms.json").exists():
        return "matrixcity"

    subdirs = [d for d in p.iterdir() if d.is_dir()]
    numeric_dirs = [d for d in subdirs if d.name.isdigit()]
    if len(numeric_dirs) >= 2:
        has_images = any(
            any(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in d.iterdir())
            for d in numeric_dirs
        )
        if has_images:
            return "whu_omvs"

    return "urbanscene"


# =============================================================================
# Per-dataset loaders
# =============================================================================

def _load_matrixcity(
    dataset_path: str,
    image_size: int = 518,
    patch_size: int = 14,
    first_k: Optional[int] = None,
    stride: int = 1,
    image_ext: str = ".jpg,.png",
    camera_ids: Optional[List[str]] = None,
    num_cameras_to_use: int = 0,
) -> Dict[str, Any]:
    """Load MatrixCity dataset (single-camera synthetic aerial).

    Directory layout::

        block_1/
        ├── 0000.png
        ├── 0001.png
        ├── ...
        └── transforms.json
    """
    p = Path(dataset_path)
    exts = [e.strip() for e in image_ext.split(",")]
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob.glob(str(p / f"*{ext}")))
    paths = sorted(paths)

    if not paths:
        raise FileNotFoundError(f"No images found in {dataset_path} with extensions {exts}")

    if first_k is not None and first_k > 0:
        paths = paths[:first_k]
    if stride > 1:
        paths = paths[::stride]

    logger.info("[MatrixCity] Found %d images in %s", len(paths), dataset_path)

    load_fn = _get_load_fn()
    images = load_fn(paths, mode="crop", image_size=image_size, patch_size=patch_size)

    gt_cameras = None
    transforms_path = p / "transforms.json"
    if transforms_path.exists():
        with open(transforms_path, "r") as f:
            gt_cameras = json.load(f)

    return {
        "images": images,
        "paths": paths,
        "resolved_folder": str(p),
        "dataset_type": "matrixcity",
        "num_cameras": 1,
        "num_frames": len(paths),
        "camera_names": ["cam_0"],
        "frame_major_paths": paths,
        "gt_cameras": gt_cameras,
        "gt_depths": None,
    }


def _load_whu_omvs(
    dataset_path: str,
    image_size: int = 518,
    patch_size: int = 14,
    first_k: Optional[int] = None,
    stride: int = 1,
    image_ext: str = ".jpg,.png",
    camera_ids: Optional[List[str]] = None,
    num_cameras_to_use: int = 0,
) -> Dict[str, Any]:
    """Load WHU-OMVS dataset (5-camera oblique rig).

    The loader supports two entry-point styles:

    A) **Area-level** – ``dataset_path`` points to an area directory that
       contains ``images/`` with camera sub-directories::

           area1/
           ├── images/
           │   ├── 1/   (001_001.png, 001_002.png, ...)
           │   ├── 2/
           │   ├── 3/
           │   ├── 4/
           │   └── 5/
           ├── depths/
           ├── cams/
           ├── masks/
           ├── normals/
           └── info/

    B) **Images-level** – ``dataset_path`` points directly to the ``images/``
       directory containing camera sub-directories::

           images/
           ├── 1/
           ├── 2/
           ├── 3/
           ├── 4/
           └── 5/

    Images are arranged in **frame-major** order:
    ``[frame0_cam0, frame0_cam1, ..., frame0_camC, frame1_cam0, ...]``
    """
    p = Path(dataset_path)

    images_dir = p
    if (p / "images").is_dir():
        images_dir = p / "images"

    camera_dirs = sorted(
        [d for d in images_dir.iterdir() if d.is_dir()],
        key=lambda d: (0, int(d.name)) if d.name.isdigit() else (1, d.name),
    )

    if not camera_dirs:
        raise FileNotFoundError(f"No camera sub-directories found in {images_dir}")

    if camera_ids is not None and len(camera_ids) > 0:
        available = {d.name: d for d in camera_dirs}
        camera_dirs = [available[cid] for cid in camera_ids if cid in available]

    if num_cameras_to_use > 0 and (camera_ids is None or len(camera_ids) == 0):
        camera_dirs = camera_dirs[:num_cameras_to_use]

    exts = [e.strip() for e in image_ext.split(",")]
    camera_image_paths: Dict[str, List[str]] = {}
    for cam_dir in camera_dirs:
        cam_paths: List[str] = []
        for ext in exts:
            cam_paths.extend(glob.glob(str(cam_dir / f"*{ext}")))
        cam_paths = sorted(cam_paths)
        camera_image_paths[cam_dir.name] = cam_paths

    num_cameras = len(camera_dirs)
    num_frames = min(len(v) for v in camera_image_paths.values())
    camera_names = [d.name for d in camera_dirs]

    if first_k is not None and first_k > 0:
        first_k_frames = first_k
        for k in camera_image_paths:
            camera_image_paths[k] = camera_image_paths[k][:first_k_frames]
        num_frames = min(len(v) for v in camera_image_paths.values())

    if stride > 1:
        for k in camera_image_paths:
            camera_image_paths[k] = camera_image_paths[k][::stride]
        num_frames = min(len(v) for v in camera_image_paths.values())

    frame_major_paths: List[str] = []
    for frame_idx in range(num_frames):
        for cam_name in camera_names:
            frame_major_paths.append(camera_image_paths[cam_name][frame_idx])

    logger.info(
        "[WHU-OMVS] cameras=%d, frames=%d, total_images=%d, camera_ids=%s",
        num_cameras, num_frames, len(frame_major_paths), camera_names,
    )

    load_fn = _get_load_fn()
    images = load_fn(
        frame_major_paths, mode="crop", image_size=image_size, patch_size=patch_size,
    )

    gt_cameras = _load_whu_omvs_gt_cameras(p, camera_names, num_frames)
    gt_depths = _load_whu_omvs_gt_depths(p, camera_names, num_frames)

    return {
        "images": images,
        "paths": frame_major_paths,
        "resolved_folder": str(p),
        "dataset_type": "whu_omvs",
        "num_cameras": num_cameras,
        "num_frames": num_frames,
        "camera_names": camera_names,
        "camera_image_paths": camera_image_paths,
        "frame_major_paths": frame_major_paths,
        "gt_cameras": gt_cameras,
        "gt_depths": gt_depths,
    }


def _load_whu_omvs_gt_cameras(
    area_path: Path,
    camera_names: List[str],
    num_frames: int,
) -> Optional[Dict[str, Any]]:
    """Load GT camera parameters from WHU-OMVS ``cams/`` or ``info/`` directories."""
    cams_dir = area_path / "cams"
    info_dir = area_path / "info"

    if cams_dir.exists():
        return _load_whu_omvs_cams_dir(cams_dir, camera_names, num_frames)

    if info_dir.exists():
        return _load_whu_omvs_info_dir(info_dir, camera_names, num_frames)

    return None


def _load_whu_omvs_cams_dir(
    cams_dir: Path,
    camera_names: List[str],
    num_frames: int,
) -> Dict[str, Any]:
    """Load per-frame camera parameters from ``cams/{cam_id}/*.txt`` files.

    Each txt file contains::

        extrinsic (4x4)
        blank line
        intrinsic (3x3)
        blank line
        depth_range (min max)
        blank line
        name mask_type h_min h_max w_min w_max
    """
    extrinsics = []
    intrinsics = []

    for cam_name in camera_names:
        cam_cams_dir = cams_dir / cam_name
        if not cam_cams_dir.exists():
            logger.warning("[WHU-OMVS] cams dir not found for camera %s", cam_name)
            continue

        cam_files = sorted(cam_cams_dir.glob("*.txt"))
        for cam_file in cam_files[:num_frames]:
            ext, int_mat = _parse_whu_omvs_cam_file(cam_file)
            extrinsics.append(ext)
            intrinsics.append(int_mat)

    if not extrinsics:
        return {"extrinsics": None, "intrinsics": None}

    return {
        "extrinsics": np.stack(extrinsics),
        "intrinsics": np.stack(intrinsics),
    }


def _parse_whu_omvs_cam_file(cam_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a single WHU-OMVS camera parameter file.

    File format (with blank lines)::
        header line
        extr_row1
        extr_row2
        extr_row3
        extr_row4
        <blank>
        intr_row1
        intr_row2
        intr_row3
        <blank>
        depth_min_max
        <blank>
        name_info

    After stripping blank lines the indices shift to 0-9.
    """
    with open(cam_file, "r") as f:
        raw_lines = f.readlines()

    non_blank = [l.strip() for l in raw_lines if l.strip()]

    ext_rows = []
    for i in range(1, 5):
        ext_rows.append([float(x) for x in non_blank[i].split()])
    extrinsic = np.array(ext_rows, dtype=np.float64)

    int_rows = []
    for i in range(5, 8):
        int_rows.append([float(x) for x in non_blank[i].split()])
    intrinsic = np.array(int_rows, dtype=np.float64)

    return extrinsic, intrinsic


def _load_whu_omvs_info_dir(
    info_dir: Path,
    camera_names: List[str],
    num_frames: int,
) -> Dict[str, Any]:
    """Load camera parameters from COLMAP-style ``info/`` directory."""
    camera_info_path = info_dir / "camera_info.txt"
    image_info_path = info_dir / "image_info.txt"

    if not camera_info_path.exists() or not image_info_path.exists():
        return {"extrinsics": None, "intrinsics": None}

    intrinsics_map = {}
    with open(camera_info_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            cam_id = parts[0]
            fx, fy = float(parts[4]), float(parts[5])
            cx, cy = float(parts[6]), float(parts[7])
            intrinsics_map[cam_id] = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ], dtype=np.float64)

    cam_name_to_id = {name: str(i + 1) for i, name in enumerate(camera_names)}

    extrinsics = []
    intrinsics = []
    with open(image_info_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            cam_id = parts[1]
            if cam_id not in cam_name_to_id.values():
                continue
            r_vals = [float(x) for x in parts[2:11]]
            t_vals = [float(x) for x in parts[11:14]]
            R = np.array(r_vals, dtype=np.float64).reshape(3, 3)
            t = np.array(t_vals, dtype=np.float64)
            ext = np.eye(4, dtype=np.float64)
            ext[:3, :3] = R
            ext[:3, 3] = t
            extrinsics.append(ext)
            if cam_id in intrinsics_map:
                intrinsics.append(intrinsics_map[cam_id])

    if not extrinsics:
        return {"extrinsics": None, "intrinsics": None}

    return {
        "extrinsics": np.stack(extrinsics),
        "intrinsics": np.stack(intrinsics) if intrinsics else None,
    }


def _load_whu_omvs_gt_depths(
    area_path: Path,
    camera_names: List[str],
    num_frames: int,
) -> Optional[Dict[str, Any]]:
    """Check for GT depth availability in WHU-OMVS ``depths/`` directory."""
    depths_dir = area_path / "depths"
    if not depths_dir.exists():
        return None

    available_cameras = []
    for cam_name in camera_names:
        cam_depth_dir = depths_dir / cam_name
        if cam_depth_dir.exists():
            depth_files = sorted(cam_depth_dir.glob("*.exr"))
            if depth_files:
                available_cameras.append(cam_name)

    if not available_cameras:
        return None

    return {
        "format": "exr",
        "available_cameras": available_cameras,
        "depths_dir": str(depths_dir),
    }


def _load_urbanscene(
    dataset_path: str,
    image_size: int = 518,
    patch_size: int = 14,
    first_k: Optional[int] = None,
    stride: int = 1,
    image_ext: str = ".jpg,.png,.JPG,.PNG,.jpeg,.JPEG",
    camera_ids: Optional[List[str]] = None,
    num_cameras_to_use: int = 0,
) -> Dict[str, Any]:
    """Load UrbanScene dataset (real drone imagery, no GT).

    Directory layout::

        PolyTech/
        ├── DJI_0001.JPG
        ├── DJI_0002.JPG
        └── ...
    """
    p = Path(dataset_path)
    exts = [e.strip() for e in image_ext.split(",")]
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob.glob(str(p / f"*{ext}")))
    paths = sorted(paths)

    if not paths:
        raise FileNotFoundError(
            f"No images found in {dataset_path} with extensions {exts}. "
            "Try --image_ext .JPG for UrbanScene datasets."
        )

    if first_k is not None and first_k > 0:
        paths = paths[:first_k]
    if stride > 1:
        paths = paths[::stride]

    logger.info("[UrbanScene] Found %d images in %s", len(paths), dataset_path)

    load_fn = _get_load_fn()
    images = load_fn(paths, mode="crop", image_size=image_size, patch_size=patch_size)

    return {
        "images": images,
        "paths": paths,
        "resolved_folder": str(p),
        "dataset_type": "urbanscene",
        "num_cameras": 1,
        "num_frames": len(paths),
        "camera_names": ["cam_0"],
        "frame_major_paths": paths,
        "gt_cameras": None,
        "gt_depths": None,
    }


# =============================================================================
# Video loading helper
# =============================================================================

def load_from_video(
    video_path: str,
    fps: int = 10,
    image_size: int = 518,
    patch_size: int = 14,
    first_k: Optional[int] = None,
    stride: int = 1,
) -> Dict[str, Any]:
    """Extract frames from a video file and preprocess them."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(os.path.dirname(video_path), f"{video_name}_frames")
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, round(src_fps / fps))

    idx, saved = 0, []
    pbar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            path = os.path.join(out_dir, f"{len(saved):06d}.jpg")
            cv2.imwrite(path, frame)
            saved.append(path)
        idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()

    paths = saved
    if first_k is not None and first_k > 0:
        paths = paths[:first_k]
    if stride > 1:
        paths = paths[::stride]

    logger.info("Extracted %d frames from video (%d total, interval=%d)", len(paths), total_frames, interval)

    load_fn = _get_load_fn()
    images = load_fn(paths, mode="crop", image_size=image_size, patch_size=patch_size)

    return {
        "images": images,
        "paths": paths,
        "resolved_folder": out_dir,
        "dataset_type": "video",
        "num_cameras": 1,
        "num_frames": len(paths),
        "camera_names": ["cam_0"],
        "frame_major_paths": paths,
        "gt_cameras": None,
        "gt_depths": None,
    }


# =============================================================================
# Main loader class
# =============================================================================

_DATASET_LOADERS = {
    "matrixcity": _load_matrixcity,
    "whu_omvs": _load_whu_omvs,
    "urbanscene": _load_urbanscene,
}


class DatasetLoader:
    """Universal dataset loader for MatrixCity, WHU-OMVS, and UrbanScene.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset root or image folder.
    dataset_type : str
        One of ``"auto"``, ``"matrixcity"``, ``"whu_omvs"``, ``"urbanscene"``.
        When ``"auto"`` (default), the type is inferred from the directory
        structure.
    image_size : int
        Target image size for preprocessing (default 518).
    patch_size : int
        Patch size for dimension alignment (default 14).
    first_k : int or None
        Only load the first *k* frames (default None = all).
    stride : int
        Frame stride for sub-sampling (default 1 = every frame).
    image_ext : str
        Comma-separated image file extensions to search for.
    camera_ids : list of str or None
        Specific camera IDs to load (WHU-OMVS only).  Overrides
        ``num_cameras_to_use``.
    num_cameras_to_use : int
        Maximum number of cameras to load (WHU-OMVS only).  0 = all.
    """

    def __init__(
        self,
        dataset_path: str,
        dataset_type: str = "auto",
        image_size: int = 518,
        patch_size: int = 14,
        first_k: Optional[int] = None,
        stride: int = 1,
        image_ext: str = ".jpg,.png,.JPG,.PNG,.jpeg,.JPEG",
        camera_ids: Optional[List[str]] = None,
        num_cameras_to_use: int = 0,
    ):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.patch_size = patch_size
        self.first_k = first_k
        self.stride = stride
        self.image_ext = image_ext
        self.camera_ids = camera_ids
        self.num_cameras_to_use = num_cameras_to_use

        if dataset_type == "auto":
            self.dataset_type = detect_dataset_type(dataset_path)
            logger.info("[DatasetLoader] Auto-detected dataset type: %s", self.dataset_type)
        else:
            self.dataset_type = dataset_type

        if self.dataset_type not in _DATASET_LOADERS:
            raise ValueError(
                f"Unknown dataset type: {self.dataset_type}. "
                f"Supported: {list(_DATASET_LOADERS.keys())}"
            )

    def load(self) -> Dict[str, Any]:
        """Load the dataset and return a unified result dictionary.

        Returns
        -------
        dict with keys:
            images : torch.Tensor
                Preprocessed image tensor.
                Shape ``[N, 3, H, W]`` for single-camera datasets,
                or ``[N*C, 3, H, W]`` for multi-camera (frame-major order).
            paths : list of str
                All image file paths in loading order.
            resolved_folder : str
                The source image folder (useful for sky-mask caching etc.).
            dataset_type : str
                One of ``"matrixcity"``, ``"whu_omvs"``, ``"urbanscene"``.
            num_cameras : int
                Number of cameras (1 for MatrixCity/UrbanScene, 5 for WHU-OMVS).
            num_frames : int
                Number of temporal frames per camera.
            camera_names : list of str
                Camera identifiers (e.g. ``["1","2","3","4","5"]`` for WHU-OMVS).
            frame_major_paths : list of str
                Image paths in frame-major order (same as ``paths`` for
                single-camera; interleaved for multi-camera).
            gt_cameras : dict or None
                Ground-truth camera parameters if available.
                WHU-OMVS: ``{"extrinsics": ndarray, "intrinsics": ndarray}``
                MatrixCity: raw ``transforms.json`` dict.
                UrbanScene: ``None``.
            gt_depths : dict or None
                Ground-truth depth metadata if available.
                WHU-OMVS: ``{"format": "exr", "available_cameras": [...], "depths_dir": str}``
                Others: ``None``.
            camera_image_paths : dict  (WHU-OMVS only)
                Per-camera image path lists ``{cam_name: [path, ...]}``.
        """
        loader_fn = _DATASET_LOADERS[self.dataset_type]
        result = loader_fn(
            dataset_path=self.dataset_path,
            image_size=self.image_size,
            patch_size=self.patch_size,
            first_k=self.first_k,
            stride=self.stride,
            image_ext=self.image_ext,
            camera_ids=self.camera_ids,
            num_cameras_to_use=self.num_cameras_to_use,
        )
        h, w = result["images"].shape[-2:]
        logger.info(
            "[%s] Loaded %d frames x %d cameras = %d images, preprocessed to %dx%d",
            result["dataset_type"],
            result["num_frames"],
            result["num_cameras"],
            len(result["paths"]),
            w, h,
        )
        return result

    def load_single_camera(
        self,
        camera_index: int = 0,
    ) -> Dict[str, Any]:
        """Load only one camera from a multi-camera dataset.

        For WHU-OMVS this is useful when the downstream model only supports
        single-camera input (e.g. LingBot-MAP streaming).

        Parameters
        ----------
        camera_index : int
            Index into ``camera_names`` (default 0 = first camera).

        Returns
        -------
        Same dict structure as :meth:`load`, but with ``num_cameras=1`` and
        images from only the selected camera.
        """
        if self.dataset_type != "whu_omvs":
            return self.load()

        result = self.load()
        cam_name = result["camera_names"][camera_index]
        cam_paths = result.get("camera_image_paths", {}).get(cam_name, [])

        if not cam_paths:
            raise ValueError(f"No images found for camera {cam_name}")

        load_fn = _get_load_fn()
        images = load_fn(
            cam_paths, mode="crop", image_size=self.image_size, patch_size=self.patch_size,
        )

        result["images"] = images
        result["paths"] = cam_paths
        result["frame_major_paths"] = cam_paths
        result["num_cameras"] = 1
        result["num_frames"] = len(cam_paths)
        result["camera_names"] = [cam_name]
        result["resolved_folder"] = str(Path(self.dataset_path))

        logger.info(
            "[WHU-OMVS] Single-camera mode: camera=%s, frames=%d",
            cam_name, len(cam_paths),
        )
        return result


# =============================================================================
# Convenience function
# =============================================================================

def load_dataset(
    dataset_path: str,
    dataset_type: str = "auto",
    image_size: int = 518,
    patch_size: int = 14,
    first_k: Optional[int] = None,
    stride: int = 1,
    image_ext: str = ".jpg,.png,.JPG,.PNG,.jpeg,.JPEG",
    camera_ids: Optional[List[str]] = None,
    num_cameras_to_use: int = 0,
    single_camera: bool = False,
    camera_index: int = 0,
) -> Dict[str, Any]:
    """One-call convenience wrapper around :class:`DatasetLoader`.

    Parameters
    ----------
    single_camera : bool
        If True, load only one camera (useful for single-camera models on
        multi-camera datasets).
    camera_index : int
        Which camera to select when ``single_camera=True``.
    """
    loader = DatasetLoader(
        dataset_path=dataset_path,
        dataset_type=dataset_type,
        image_size=image_size,
        patch_size=patch_size,
        first_k=first_k,
        stride=stride,
        image_ext=image_ext,
        camera_ids=camera_ids,
        num_cameras_to_use=num_cameras_to_use,
    )
    if single_camera:
        return loader.load_single_camera(camera_index=camera_index)
    return loader.load()
