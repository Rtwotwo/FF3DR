from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class PointCloudReconTotals:
    pred_count: int = 0
    gt_count: int = 0
    pred_matched: int = 0
    gt_matched: int = 0
    pred_to_gt_dist_sum: float = 0.0
    gt_to_pred_dist_sum: float = 0.0

    def add(self, other: "PointCloudReconTotals") -> None:
        self.pred_count += other.pred_count
        self.gt_count += other.gt_count
        self.pred_matched += other.pred_matched
        self.gt_matched += other.gt_matched
        self.pred_to_gt_dist_sum += other.pred_to_gt_dist_sum
        self.gt_to_pred_dist_sum += other.gt_to_pred_dist_sum


def load_ply_binary(path: Path) -> np.ndarray:
    with open(str(path), "rb") as f:
        line = f.readline().decode("ascii").strip()
        if line != "ply":
            raise ValueError("Not a PLY file: {}".format(path))
        format_line = f.readline().decode("ascii").strip()
        vertex_count = 0
        has_rgb = False
        property_dtype_map = {}
        prop_order = []
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                dtype_name = parts[1]
                prop_name = parts[2]
                if dtype_name == "float":
                    property_dtype_map[prop_name] = np.float32
                    prop_order.append((prop_name, np.float32))
                elif dtype_name == "uchar":
                    property_dtype_map[prop_name] = np.uint8
                    prop_order.append((prop_name, np.uint8))
                    if prop_name in ("red", "green", "blue", "alpha", "r", "g", "b", "a"):
                        has_rgb = True
                elif dtype_name == "double":
                    property_dtype_map[prop_name] = np.float64
                    prop_order.append((prop_name, np.float64))
                elif dtype_name == "int":
                    property_dtype_map[prop_name] = np.int32
                    prop_order.append((prop_name, np.int32))
            elif line == "end_header":
                break

        if vertex_count == 0:
            return np.zeros((0, 3), dtype=np.float32)

        is_binary = "binary" in format_line

        if is_binary:
            float_count = sum(1 for _, dt in prop_order if dt in (np.float32, np.float64, np.int32))
            uint_count = sum(1 for _, dt in prop_order if dt == np.uint8)
            bytes_per_vertex = float_count * 4 + uint_count
            raw = f.read(vertex_count * bytes_per_vertex)

            xyz = np.zeros((vertex_count, 3), dtype=np.float32)
            offset = 0
            float_idx = 0
            for prop_name, dt in prop_order:
                if dt == np.float32:
                    col = np.frombuffer(raw, dtype=np.float32, count=vertex_count, offset=offset)
                    if prop_name in ("x", "y", "z") and float_idx < 3:
                        xyz[:, float_idx] = col
                        float_idx += 1
                    offset += 4 * vertex_count
                elif dt == np.float64:
                    col = np.frombuffer(raw, dtype=np.float64, count=vertex_count, offset=offset)
                    if prop_name in ("x", "y", "z") and float_idx < 3:
                        xyz[:, float_idx] = col.astype(np.float32)
                        float_idx += 1
                    offset += 8 * vertex_count
                elif dt == np.uint8:
                    offset += 1 * vertex_count
                elif dt == np.int32:
                    offset += 4 * vertex_count
            return xyz
        else:
            points = []
            for _ in range(vertex_count):
                line = f.readline().decode("ascii").strip()
                vals = line.split()
                x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
                points.append([x, y, z])
            return np.array(points, dtype=np.float32)


def load_ply(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("PLY file not found: {}".format(path))
    with open(str(path), "rb") as f:
        line = f.readline().decode("ascii").strip()
    if line != "ply":
        raise ValueError("Not a PLY file: {}".format(path))
    with open(str(path), "rb") as f:
        header_lines = []
        while True:
            l = f.readline().decode("ascii").strip()
            header_lines.append(l)
            if l == "end_header":
                break
        peek = f.read(1)
    is_binary = any("binary" in hl for hl in header_lines)
    if is_binary:
        pts = load_ply_binary(path)
    else:
        data = np.loadtxt(str(path), skiprows=len(header_lines))
        pts = data[:, :3].astype(np.float32)
    finite_mask = np.all(np.isfinite(pts), axis=1)
    return pts[finite_mask]


def _batch_nearest_neighbor_distance(
    source: np.ndarray,
    target: np.ndarray,
    batch_size: int = 500_000,
) -> np.ndarray:
    if source.shape[0] == 0 or target.shape[0] == 0:
        return np.full(source.shape[0], np.inf, dtype=np.float32)

    from scipy.spatial import cKDTree

    tree = cKDTree(target)
    dists = np.empty(source.shape[0], dtype=np.float32)
    for start in range(0, source.shape[0], batch_size):
        end = min(start + batch_size, source.shape[0])
        d, _ = tree.query(source[start:end], k=1, workers=-1)
        dists[start:end] = d
    return dists


def compute_recon_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    threshold: float = 0.5,
    batch_size: int = 500_000,
) -> Dict[str, float]:
    pred_points = np.asarray(pred_points, dtype=np.float64).reshape(-1, 3)
    gt_points = np.asarray(gt_points, dtype=np.float64).reshape(-1, 3)

    pred_finite_mask = np.all(np.isfinite(pred_points), axis=1)
    gt_finite_mask = np.all(np.isfinite(gt_points), axis=1)
    n_pred_invalid = int((~pred_finite_mask).sum())
    n_gt_invalid = int((~gt_finite_mask).sum())
    if n_pred_invalid > 0:
        import logging
        logging.getLogger(__name__).warning(
            "Filtered %d / %d non-finite pred points (NaN/Inf)", n_pred_invalid, pred_points.shape[0])
        pred_points = pred_points[pred_finite_mask]
    if n_gt_invalid > 0:
        import logging
        logging.getLogger(__name__).warning(
            "Filtered %d / %d non-finite GT points (NaN/Inf)", n_gt_invalid, gt_points.shape[0])
        gt_points = gt_points[gt_finite_mask]

    if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": float("inf"),
            "completeness": 0.0,
            "pred_count": int(pred_points.shape[0]),
            "gt_count": int(gt_points.shape[0]),
            "threshold": float(threshold),
        }

    pred_to_gt_dists = _batch_nearest_neighbor_distance(pred_points, gt_points, batch_size)
    gt_to_pred_dists = _batch_nearest_neighbor_distance(gt_points, pred_points, batch_size)

    pred_matched = int(np.sum(pred_to_gt_dists <= threshold))
    gt_matched = int(np.sum(gt_to_pred_dists <= threshold))

    precision = pred_matched / pred_points.shape[0]
    recall = gt_matched / gt_points.shape[0]
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = float(np.mean(pred_to_gt_dists))
    completeness = recall

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "completeness": completeness,
        "pred_count": int(pred_points.shape[0]),
        "gt_count": int(gt_points.shape[0]),
        "pred_matched": pred_matched,
        "gt_matched": gt_matched,
        "threshold": float(threshold),
    }


def compute_recon_metrics_multi_threshold(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    thresholds: Tuple[float, ...] = (0.2, 0.5, 1.0, 2.0),
    batch_size: int = 500_000,
) -> Dict[str, Dict[str, float]]:
    pred_points = np.asarray(pred_points, dtype=np.float64).reshape(-1, 3)
    gt_points = np.asarray(gt_points, dtype=np.float64).reshape(-1, 3)

    pred_finite_mask = np.all(np.isfinite(pred_points), axis=1)
    gt_finite_mask = np.all(np.isfinite(gt_points), axis=1)
    n_pred_invalid = int((~pred_finite_mask).sum())
    n_gt_invalid = int((~gt_finite_mask).sum())
    if n_pred_invalid > 0:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "Filtered %d / %d non-finite pred points (NaN/Inf)", n_pred_invalid, pred_points.shape[0])
        pred_points = pred_points[pred_finite_mask]
    if n_gt_invalid > 0:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "Filtered %d / %d non-finite GT points (NaN/Inf)", n_gt_invalid, gt_points.shape[0])
        gt_points = gt_points[gt_finite_mask]

    if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
        return {
            "tau_{:.2f}".format(t): {
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "accuracy": float("inf"), "completeness": 0.0,
                "pred_count": 0, "gt_count": 0, "threshold": t,
            }
            for t in thresholds
        }

    from scipy.spatial import cKDTree

    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(pred_points)

    pred_to_gt_dists = np.empty(pred_points.shape[0], dtype=np.float32)
    for start in range(0, pred_points.shape[0], batch_size):
        end = min(start + batch_size, pred_points.shape[0])
        d, _ = gt_tree.query(pred_points[start:end], k=1, workers=-1)
        pred_to_gt_dists[start:end] = d

    gt_to_pred_dists = np.empty(gt_points.shape[0], dtype=np.float32)
    for start in range(0, gt_points.shape[0], batch_size):
        end = min(start + batch_size, gt_points.shape[0])
        d, _ = pred_tree.query(gt_points[start:end], k=1, workers=-1)
        gt_to_pred_dists[start:end] = d

    accuracy = float(np.mean(pred_to_gt_dists))
    results = {}
    for t in thresholds:
        pred_matched = int(np.sum(pred_to_gt_dists <= t))
        gt_matched = int(np.sum(gt_to_pred_dists <= t))
        precision = pred_matched / pred_points.shape[0]
        recall = gt_matched / gt_points.shape[0]
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results["tau_{:.2f}".format(t)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "completeness": recall,
            "pred_count": int(pred_points.shape[0]),
            "gt_count": int(gt_points.shape[0]),
            "pred_matched": pred_matched,
            "gt_matched": gt_matched,
            "threshold": float(t),
        }
    return results


def unproject_depth_to_points(
    depth_map: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    depth_scale: float = 1.0,
    mask: Optional[np.ndarray] = None,
    stride: int = 1,
) -> np.ndarray:
    depth_map = np.asarray(depth_map, dtype=np.float32)
    h, w = depth_map.shape

    if mask is not None:
        valid = (mask > 0.5) & np.isfinite(depth_map) & (depth_map > 1e-6)
    else:
        valid = np.isfinite(depth_map) & (depth_map > 1e-6)

    if stride > 1:
        valid[::stride, :] = False
        valid[:, ::stride] = False

    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    zs = depth_map[ys, xs] * depth_scale

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    z_cam = zs

    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]

    finite_mask = np.all(np.isfinite(pts_world), axis=1)
    pts_world = pts_world[finite_mask]

    return pts_world.astype(np.float32)


class PointCloudReconAccumulator:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.totals = PointCloudReconTotals()
        self._pred_points_list = []
        self._gt_points_list = []
        self._finalized = False
        self._cached_result = None

    def update(
        self,
        pred_points: np.ndarray,
        gt_points: np.ndarray,
    ) -> None:
        self._pred_points_list.append(np.asarray(pred_points, dtype=np.float32).reshape(-1, 3))
        self._gt_points_list.append(np.asarray(gt_points, dtype=np.float32).reshape(-1, 3))
        self._finalized = False

    def finalize(self, batch_size: int = 500_000) -> Dict[str, float]:
        if self._finalized and self._cached_result is not None:
            return self._cached_result

        pred_all = np.concatenate(self._pred_points_list, axis=0) if self._pred_points_list else np.zeros((0, 3), dtype=np.float32)
        gt_all = np.concatenate(self._gt_points_list, axis=0) if self._gt_points_list else np.zeros((0, 3), dtype=np.float32)

        result = compute_recon_metrics(pred_all, gt_all, self.threshold, batch_size)
        self._finalized = True
        self._cached_result = result
        return result

    def finalize_multi_threshold(
        self,
        thresholds: Tuple[float, ...] = (0.2, 0.5, 1.0, 2.0),
        batch_size: int = 500_000,
    ) -> Dict[str, Dict[str, float]]:
        pred_all = np.concatenate(self._pred_points_list, axis=0) if self._pred_points_list else np.zeros((0, 3), dtype=np.float32)
        gt_all = np.concatenate(self._gt_points_list, axis=0) if self._gt_points_list else np.zeros((0, 3), dtype=np.float32)
        return compute_recon_metrics_multi_threshold(pred_all, gt_all, thresholds, batch_size)
