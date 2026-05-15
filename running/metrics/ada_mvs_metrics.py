from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class AdaMVSMetricTotals:
    all_valid_count: int = 0
    within_count: int = 0
    mae_sum: float = 0.0
    mse_sum: float = 0.0
    pag_02_count: int = 0
    pag_04_count: int = 0
    pag_06_count: int = 0
    abs_rel_sum: float = 0.0
    sq_rel_sum: float = 0.0
    rmse_log_sq_sum: float = 0.0
    log10_sum: float = 0.0
    silog_log_sum: float = 0.0
    silog_log_sq_sum: float = 0.0
    delta1_hits: int = 0
    delta2_hits: int = 0
    delta3_hits: int = 0

    def add(self, other: AdaMVSMetricTotals) -> None:
        self.all_valid_count += other.all_valid_count
        self.within_count += other.within_count
        self.mae_sum += other.mae_sum
        self.mse_sum += other.mse_sum
        self.pag_02_count += other.pag_02_count
        self.pag_04_count += other.pag_04_count
        self.pag_06_count += other.pag_06_count
        self.abs_rel_sum += other.abs_rel_sum
        self.sq_rel_sum += other.sq_rel_sum
        self.rmse_log_sq_sum += other.rmse_log_sq_sum
        self.log10_sum += other.log10_sum
        self.silog_log_sum += other.silog_log_sum
        self.silog_log_sq_sum += other.silog_log_sq_sum
        self.delta1_hits += other.delta1_hits
        self.delta2_hits += other.delta2_hits
        self.delta3_hits += other.delta3_hits


@dataclass
class AdaMVSMetricAccumulator:
    outlier_threshold: float = 20.0
    align_mode: str = "none"
    totals: AdaMVSMetricTotals = field(default_factory=AdaMVSMetricTotals)

    def _align(self, pred_depth: np.ndarray, gt_depth: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        if self.align_mode != "median":
            return pred_depth
        valid = np.isfinite(pred_depth) & np.isfinite(gt_depth) & (pred_depth > 1e-8) & (gt_depth > 1e-8)
        if mask is not None:
            valid &= np.asarray(mask) > 0.5
        if not np.any(valid):
            return pred_depth
        scale = float(np.median(gt_depth[valid])) / float(np.median(pred_depth[valid]))
        if not np.isfinite(scale) or abs(scale) < 1e-8:
            return pred_depth
        return pred_depth * scale

    def update(self, pred_depth, gt_depth, mask: Optional[np.ndarray] = None) -> None:
        pred_depth = np.asarray(pred_depth, dtype=np.float64)
        gt_depth = np.asarray(gt_depth, dtype=np.float64)
        if mask is not None:
            mask = np.asarray(mask)

        pred_depth = self._align(pred_depth, gt_depth, mask)

        valid = np.isfinite(pred_depth) & np.isfinite(gt_depth)
        valid &= (pred_depth > 1e-8) & (gt_depth > 1e-8)
        if mask is not None:
            valid &= mask > 0.5

        all_valid_count = int(valid.sum())
        if all_valid_count == 0:
            return

        abs_diff = np.abs(pred_depth - gt_depth)

        within_threshold = valid & (abs_diff <= self.outlier_threshold)
        within_count = int(within_threshold.sum())

        diff_all = abs_diff[valid]
        pred_all = pred_depth[valid]
        gt_all = gt_depth[valid]

        self.totals.all_valid_count += all_valid_count

        self.totals.pag_02_count += int((diff_all < 0.2).sum())
        self.totals.pag_04_count += int((diff_all < 0.4).sum())
        self.totals.pag_06_count += int((diff_all < 0.6).sum())

        gt_safe_all = np.where(gt_all > 1e-8, gt_all, 1e-8)
        ratio = np.maximum(pred_all / gt_safe_all, gt_safe_all / pred_all)
        self.totals.delta1_hits += int((ratio < 1.25).sum())
        self.totals.delta2_hits += int((ratio < 1.25 ** 2).sum())
        self.totals.delta3_hits += int((ratio < 1.25 ** 3).sum())

        if within_count == 0:
            return

        diff_within = abs_diff[within_threshold]
        pred_within = pred_depth[within_threshold]
        gt_within = gt_depth[within_threshold]

        self.totals.within_count += within_count
        self.totals.mae_sum += float(diff_within.sum())
        self.totals.mse_sum += float(np.square(diff_within).sum())

        gt_safe = np.where(gt_within > 1e-8, gt_within, 1e-8)
        self.totals.abs_rel_sum += float((diff_within / gt_safe).sum())
        self.totals.sq_rel_sum += float(np.square(diff_within / gt_safe).sum())

        log_diff = np.log(pred_within) - np.log(gt_safe)
        self.totals.rmse_log_sq_sum += float(np.square(log_diff).sum())
        self.totals.log10_sum += float(np.abs(log_diff / np.log(10.0)).sum())
        self.totals.silog_log_sum += float(log_diff.sum())
        self.totals.silog_log_sq_sum += float(np.square(log_diff).sum())

    def merge(self, other: AdaMVSMetricAccumulator) -> None:
        self.totals.add(other.totals)

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
                "abs_rel": float("nan"),
                "sq_rel": float("nan"),
                "rmse_log": float("nan"),
                "log10": float("nan"),
                "silog": float("nan"),
                "delta1": float("nan"),
                "delta2": float("nan"),
                "delta3": float("nan"),
                "valid_count": 0,
                "outlier_count": 0,
            }

        pag_denom = all_count
        mae_denom = within_count if within_count > 0 else all_count

        mean_log_diff = self.totals.silog_log_sum / mae_denom
        mean_log_diff_sq = self.totals.silog_log_sq_sum / mae_denom
        silog = math.sqrt(max(mean_log_diff_sq - mean_log_diff * mean_log_diff, 0.0)) * 100.0

        return {
            "MAE": self.totals.mae_sum / mae_denom,
            "RMSE": math.sqrt(self.totals.mse_sum / mae_denom),
            "PAG_0.2m": self.totals.pag_02_count / pag_denom * 100.0,
            "PAG_0.4m": self.totals.pag_04_count / pag_denom * 100.0,
            "PAG_0.6m": self.totals.pag_06_count / pag_denom * 100.0,
            "abs_rel": self.totals.abs_rel_sum / mae_denom,
            "sq_rel": self.totals.sq_rel_sum / mae_denom,
            "rmse_log": math.sqrt(self.totals.rmse_log_sq_sum / mae_denom),
            "log10": self.totals.log10_sum / mae_denom,
            "silog": silog,
            "delta1": self.totals.delta1_hits / pag_denom,
            "delta2": self.totals.delta2_hits / pag_denom,
            "delta3": self.totals.delta3_hits / pag_denom,
            "valid_count": all_count,
            "outlier_count": all_count - within_count,
        }
