from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

EPS = 1e-8


def to_numpy(array) -> np.ndarray:
    if array is None:
        return None
    if isinstance(array, np.ndarray):
        return array
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def build_valid_mask(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    mask: Optional[np.ndarray] = None,
    min_depth: float = EPS,
) -> np.ndarray:
    pred_depth = np.asarray(pred_depth)
    gt_depth = np.asarray(gt_depth)
    valid = np.isfinite(pred_depth) & np.isfinite(gt_depth)
    valid &= pred_depth > min_depth
    valid &= gt_depth > min_depth
    if mask is not None:
        valid &= np.asarray(mask) > 0.5
    return valid


def align_median_scale(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    mask: Optional[np.ndarray] = None,
    min_depth: float = EPS,
) -> Tuple[np.ndarray, float]:
    valid = build_valid_mask(pred_depth, gt_depth, mask=mask, min_depth=min_depth)
    if not np.any(valid):
        return np.asarray(pred_depth), 1.0

    pred_valid = np.asarray(pred_depth)[valid]
    gt_valid = np.asarray(gt_depth)[valid]
    pred_median = float(np.median(pred_valid))
    gt_median = float(np.median(gt_valid))
    if not np.isfinite(pred_median) or abs(pred_median) < min_depth:
        return np.asarray(pred_depth), 1.0
    return np.asarray(pred_depth) * (gt_median / pred_median), gt_median / pred_median


@dataclass
class MetricTotals:
    valid_count: int = 0
    abs_rel_sum: float = 0.0
    sq_rel_sum: float = 0.0
    rmse_sq_sum: float = 0.0
    rmse_log_sq_sum: float = 0.0
    log10_sum: float = 0.0
    delta1_hits: int = 0
    delta2_hits: int = 0
    delta3_hits: int = 0
    silog_log_sum: float = 0.0
    silog_log_sq_sum: float = 0.0

    def add(self, other: "MetricTotals") -> None:
        self.valid_count += other.valid_count
        self.abs_rel_sum += other.abs_rel_sum
        self.sq_rel_sum += other.sq_rel_sum
        self.rmse_sq_sum += other.rmse_sq_sum
        self.rmse_log_sq_sum += other.rmse_log_sq_sum
        self.log10_sum += other.log10_sum
        self.delta1_hits += other.delta1_hits
        self.delta2_hits += other.delta2_hits
        self.delta3_hits += other.delta3_hits
        self.silog_log_sum += other.silog_log_sum
        self.silog_log_sq_sum += other.silog_log_sq_sum
