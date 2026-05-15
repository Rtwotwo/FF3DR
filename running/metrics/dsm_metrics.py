from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CameraParams:
    camera_id: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class ImageParams:
    image_id: int
    camera_id: int
    Rwc: np.ndarray
    twc: np.ndarray
    min_depth: float
    max_depth: float
    name: str


def load_camera_params(path: Path) -> Dict[int, CameraParams]:
    cams = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            cid = int(elems[0])
            w = int(elems[1])
            h = int(elems[2])
            fx = float(elems[4])
            fy = float(elems[5])
            cx = float(elems[6])
            cy = float(elems[7])
            cams[cid] = CameraParams(
                camera_id=cid, width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy
            )
    return cams


def load_image_params(path: Path) -> Dict[int, ImageParams]:
    images = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            iid = int(elems[0])
            cid = int(elems[1])
            R = np.array([float(x) for x in elems[2:11]], dtype=np.float64).reshape(
                3, 3
            )
            t = np.array([float(x) for x in elems[11:14]], dtype=np.float64)
            min_d = float(elems[14])
            max_d = float(elems[15])
            name = elems[16]
            images[iid] = ImageParams(
                image_id=iid,
                camera_id=cid,
                Rwc=R,
                twc=t,
                min_depth=min_d,
                max_depth=max_d,
                name=name,
            )
    return images


@dataclass
class DSMGrid:
    x_origin: float
    y_origin: float
    gsd: float
    width: int
    height: int
    data: np.ndarray

    def world_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        col = (x - self.x_origin) / self.gsd
        row = (self.y_origin - y) / self.gsd
        return row, col

    def pixel_to_world(self, row: int, col: int) -> Tuple[float, float]:
        x = self.x_origin + col * self.gsd
        y = self.y_origin - row * self.gsd
        return x, y


def load_dsm_tif(path: Path) -> DSMGrid:
    with open(path, "rb") as f:
        byte_order = f.read(2)
        little = byte_order == b"II"
        fmt = "<" if little else ">"

        magic = struct.unpack(fmt + "H", f.read(2))[0]
        assert magic == 42, f"Not a TIFF file: magic={magic}"

        ifd_offset = struct.unpack(fmt + "I", f.read(4))[0]
        f.seek(ifd_offset)
        num_entries = struct.unpack(fmt + "H", f.read(2))[0]

        width = height = bits_per_sample = sample_format = None
        strip_offsets = None

        for _ in range(num_entries):
            tag = struct.unpack(fmt + "H", f.read(2))[0]
            field_type = struct.unpack(fmt + "H", f.read(2))[0]
            count = struct.unpack(fmt + "I", f.read(4))[0]

            if field_type == 3:
                value = struct.unpack(fmt + "H", f.read(2))[0]
                f.read(2)
            elif field_type == 4:
                value = struct.unpack(fmt + "I", f.read(4))[0]
            elif field_type == 5:
                value_offset = struct.unpack(fmt + "I", f.read(4))[0]
                pos = f.tell()
                f.seek(value_offset)
                num_r = struct.unpack(fmt + "I", f.read(4))[0]
                den_r = struct.unpack(fmt + "I", f.read(4))[0]
                value = num_r / den_r if den_r != 0 else 0
                f.seek(pos)
            else:
                value = struct.unpack(fmt + "I", f.read(4))[0]

            if tag == 256:
                width = value
            elif tag == 257:
                height = value
            elif tag == 258:
                bits_per_sample = value
            elif tag == 273:
                strip_offsets = value
            elif tag == 339:
                sample_format = value

    assert width is not None and height is not None
    assert bits_per_sample == 32 and sample_format == 3

    tfw_path = path.with_suffix(".tfw")
    gsd = 0.2
    x_origin = 0.0
    y_origin = 0.0
    if tfw_path.exists():
        with open(tfw_path, "r") as f:
            lines = f.readlines()
            gsd = abs(float(lines[0].strip()))
            x_origin = float(lines[4].strip())
            y_origin = float(lines[5].strip())

    with open(path, "rb") as f:
        f.seek(strip_offsets)
        data = np.frombuffer(
            f.read(width * height * 4), dtype=np.float32
        ).reshape(height, width)

    return DSMGrid(
        x_origin=x_origin,
        y_origin=y_origin,
        gsd=gsd,
        width=width,
        height=height,
        data=data.copy(),
    )


def load_dsm_world_file(tfw_path: Path) -> Tuple[float, float, float]:
    gsd = 0.2
    x_origin = 0.0
    y_origin = 0.0
    if tfw_path.exists():
        with open(tfw_path, "r") as f:
            lines = f.readlines()
            gsd = abs(float(lines[0].strip()))
            x_origin = float(lines[4].strip())
            y_origin = float(lines[5].strip())
    return gsd, x_origin, y_origin


def unproject_depth_to_world(
    depth_map: np.ndarray,
    cam: CameraParams,
    img: ImageParams,
    downsample_factor: float = 1.0,
) -> np.ndarray:
    h, w = depth_map.shape

    if downsample_factor != 1.0:
        fx = cam.fx / downsample_factor
        fy = cam.fy / downsample_factor
        cx = cam.cx / downsample_factor
        cy = cam.cy / downsample_factor
    else:
        fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy

    u_coords = np.arange(w, dtype=np.float64)
    v_coords = np.arange(h, dtype=np.float64)
    uu, vv = np.meshgrid(u_coords, v_coords)

    depth = depth_map.astype(np.float64)

    O_xrightyup = np.array(
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64
    )
    Rwc_down = img.Rwc @ O_xrightyup

    x_cam = (uu - cx) * depth / fx
    y_cam = (vv - cy) * depth / fy
    z_cam = depth

    cam_coords = np.stack([x_cam, y_cam, z_cam], axis=-1)
    world_coords = np.einsum('ij,hwj->hwi', Rwc_down, cam_coords)
    world_coords[:, :, 0] += img.twc[0]
    world_coords[:, :, 1] += img.twc[1]
    world_coords[:, :, 2] += img.twc[2]

    return world_coords


def depth_to_dsm(
    depth_map: np.ndarray,
    cam: CameraParams,
    img: ImageParams,
    dsm_grid: DSMGrid,
    downsample_factor: float = 1.0,
) -> np.ndarray:
    P_world = unproject_depth_to_world(depth_map, cam, img, downsample_factor)

    sum_dsm = np.zeros(
        (dsm_grid.height, dsm_grid.width), dtype=np.float64
    )
    count_map = np.zeros(
        (dsm_grid.height, dsm_grid.width), dtype=np.float64
    )

    valid = np.isfinite(P_world[:, :, 0]) & np.isfinite(P_world[:, :, 1]) & np.isfinite(P_world[:, :, 2])

    x_world = P_world[:, :, 0][valid]
    y_world = P_world[:, :, 1][valid]
    z_world = P_world[:, :, 2][valid]

    rows_f = (dsm_grid.y_origin - y_world) / dsm_grid.gsd
    cols_f = (x_world - dsm_grid.x_origin) / dsm_grid.gsd

    rows = np.round(rows_f).astype(np.int64)
    cols = np.round(cols_f).astype(np.int64)

    in_bounds = (
        (rows >= 0)
        & (rows < dsm_grid.height)
        & (cols >= 0)
        & (cols < dsm_grid.width)
    )

    rows = rows[in_bounds]
    cols = cols[in_bounds]
    z_world = z_world[in_bounds]

    np.add.at(sum_dsm, (rows, cols), z_world)
    np.add.at(count_map, (rows, cols), 1.0)

    valid_dsm = count_map > 0
    pred_dsm = np.full(
        (dsm_grid.height, dsm_grid.width), np.nan, dtype=np.float64
    )
    pred_dsm[valid_dsm] = sum_dsm[valid_dsm] / count_map[valid_dsm]

    return pred_dsm


def compute_dsm_metrics_against_gt(
    pred_dsm: np.ndarray,
    gt_dsm: np.ndarray,
    outlier_threshold: float = 20.0,
    pag_thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    if pag_thresholds is None:
        pag_thresholds = [0.2, 0.4, 0.6]

    valid = np.isfinite(pred_dsm) & np.isfinite(gt_dsm)
    all_valid_count = int(valid.sum())
    if all_valid_count == 0:
        result = {
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "valid_count": 0,
            "outlier_count": 0,
        }
        for alpha in pag_thresholds:
            result["PAG_{:.1f}m".format(alpha)] = float("nan")
        return result

    abs_error = np.abs(pred_dsm[valid] - gt_dsm[valid])

    within_threshold = abs_error <= outlier_threshold
    within_count = int(within_threshold.sum())

    mae_denom = within_count if within_count > 0 else all_valid_count

    if within_count > 0:
        mae = float(abs_error[within_threshold].sum()) / mae_denom
        mse = float(np.square(abs_error[within_threshold]).sum()) / mae_denom
        rmse = math.sqrt(mse)
    else:
        mae = float(abs_error.sum()) / mae_denom
        mse = float(np.square(abs_error).sum()) / mae_denom
        rmse = math.sqrt(mse)

    result = {
        "MAE": mae,
        "RMSE": rmse,
        "valid_count": all_valid_count,
        "outlier_count": all_valid_count - within_count,
    }

    for alpha in pag_thresholds:
        m_alpha = int((abs_error < alpha).sum())
        pag = (m_alpha / all_valid_count) * 100.0
        result["PAG_{:.1f}m".format(alpha)] = pag

    return result


def compute_elevation_error_per_pixel(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    cam: CameraParams,
    img: ImageParams,
    downsample_factor: float = 1.0,
    max_pixels: int = 500000,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = gt_depth.shape
    total_pixels = h * w
    extra_ds = 1.0
    if total_pixels > max_pixels:
        extra_ds = math.sqrt(total_pixels / max_pixels)
    effective_ds = downsample_factor * extra_ds

    if extra_ds > 1.0:
        new_h = int(round(h / extra_ds))
        new_w = int(round(w / extra_ds))
        import cv2 as _cv2
        pred_depth = _cv2.resize(pred_depth, (new_w, new_h), interpolation=_cv2.INTER_LINEAR)
        gt_depth = _cv2.resize(gt_depth, (new_w, new_h), interpolation=_cv2.INTER_LINEAR)

    P_world_pred = unproject_depth_to_world(pred_depth, cam, img, effective_ds)
    P_world_gt = unproject_depth_to_world(gt_depth, cam, img, effective_ds)

    valid = (
        np.isfinite(P_world_pred[:, :, 2])
        & np.isfinite(P_world_gt[:, :, 2])
        & (gt_depth > 1e-8)
    )

    elevation_error = np.full_like(gt_depth, np.nan, dtype=np.float64)
    elevation_error[valid] = P_world_pred[valid, 2] - P_world_gt[valid, 2]

    return elevation_error, valid


@dataclass
class DSMMetricTotals:
    all_valid_count: int = 0
    within_count: int = 0
    mae_sum: float = 0.0
    mse_sum: float = 0.0
    pag_02_count: int = 0
    pag_04_count: int = 0
    pag_06_count: int = 0


@dataclass
class DSMMetricAccumulator:
    outlier_threshold: float = 20.0
    totals: DSMMetricTotals = field(default_factory=DSMMetricTotals)

    def update(
        self,
        elevation_error: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        elevation_error = np.asarray(elevation_error, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool)

        abs_error = np.abs(elevation_error[valid])
        all_valid_count = int(abs_error.shape[0])
        if all_valid_count == 0:
            return

        within_threshold = abs_error <= self.outlier_threshold
        within_count = int(within_threshold.sum())

        self.totals.all_valid_count += all_valid_count

        self.totals.pag_02_count += int((abs_error < 0.2).sum())
        self.totals.pag_04_count += int((abs_error < 0.4).sum())
        self.totals.pag_06_count += int((abs_error < 0.6).sum())

        if within_count > 0:
            diff_within = abs_error[within_threshold]
            self.totals.within_count += within_count
            self.totals.mae_sum += float(diff_within.sum())
            self.totals.mse_sum += float(np.square(diff_within).sum())

    def merge(self, other: DSMMetricAccumulator) -> None:
        self.totals.all_valid_count += other.totals.all_valid_count
        self.totals.within_count += other.totals.within_count
        self.totals.mae_sum += other.totals.mae_sum
        self.totals.mse_sum += other.totals.mse_sum
        self.totals.pag_02_count += other.totals.pag_02_count
        self.totals.pag_04_count += other.totals.pag_04_count
        self.totals.pag_06_count += other.totals.pag_06_count

    def finalize(self) -> Dict[str, float]:
        all_count = self.totals.all_valid_count
        within_count = self.totals.within_count
        if all_count == 0:
            return {
                "MAE": float("nan"),
                "RMSE": float("nan"),
                "PAG_0.2m": float("nan"),
                "PAG_0.4m": float("nan"),
                "PAG_0.6m": float("nan"),
                "valid_count": 0,
                "outlier_count": 0,
            }

        mae_denom = within_count if within_count > 0 else all_count

        return {
            "MAE": self.totals.mae_sum / mae_denom,
            "RMSE": math.sqrt(self.totals.mse_sum / mae_denom),
            "PAG_0.2m": self.totals.pag_02_count / all_count * 100.0,
            "PAG_0.4m": self.totals.pag_04_count / all_count * 100.0,
            "PAG_0.6m": self.totals.pag_06_count / all_count * 100.0,
            "valid_count": all_count,
            "outlier_count": all_count - within_count,
        }


def build_image_name_to_params(
    image_params: Dict[int, ImageParams],
    camera_params: Dict[int, CameraParams],
) -> Dict[str, Tuple[CameraParams, ImageParams]]:
    name_to_params = {}
    for img in image_params.values():
        cam = camera_params.get(img.camera_id)
        if cam is not None:
            name_to_params[img.name] = (cam, img)
    return name_to_params
