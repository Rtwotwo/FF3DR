from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np

from .abs_rel import accumulate_abs_rel
from .common import MetricTotals, align_median_scale, to_numpy
from .delta1 import accumulate_delta1
from .delta2 import accumulate_delta2
from .delta3 import accumulate_delta3
from .log10 import accumulate_log10
from .rmse import accumulate_rmse
from .rmse_log import accumulate_rmse_log
from .silog import accumulate_silog_terms
from .sq_rel import accumulate_sq_rel


@dataclass
class DepthMetricAccumulator:
    align_mode: str = "none"
    totals: MetricTotals = field(default_factory=MetricTotals)

    def update(self, pred_depth, gt_depth, mask: Optional[np.ndarray] = None) -> None:
        pred_depth = to_numpy(pred_depth)
        gt_depth = to_numpy(gt_depth)
        mask = None if mask is None else to_numpy(mask)

        if self.align_mode == "median":
            pred_depth, _ = align_median_scale(pred_depth, gt_depth, mask=mask)

        abs_rel_sum, valid_count = accumulate_abs_rel(pred_depth, gt_depth, mask=mask)
        if valid_count == 0:
            return

        sq_rel_sum, _ = accumulate_sq_rel(pred_depth, gt_depth, mask=mask)
        rmse_sq_sum, _ = accumulate_rmse(pred_depth, gt_depth, mask=mask)
        rmse_log_sq_sum, _ = accumulate_rmse_log(pred_depth, gt_depth, mask=mask)
        log10_sum, _ = accumulate_log10(pred_depth, gt_depth, mask=mask)
        delta1_hits, _ = accumulate_delta1(pred_depth, gt_depth, mask=mask)
        delta2_hits, _ = accumulate_delta2(pred_depth, gt_depth, mask=mask)
        delta3_hits, _ = accumulate_delta3(pred_depth, gt_depth, mask=mask)
        silog_log_sum, silog_log_sq_sum, _ = accumulate_silog_terms(pred_depth, gt_depth, mask=mask)

        self.totals.valid_count += valid_count
        self.totals.abs_rel_sum += abs_rel_sum
        self.totals.sq_rel_sum += sq_rel_sum
        self.totals.rmse_sq_sum += rmse_sq_sum
        self.totals.rmse_log_sq_sum += rmse_log_sq_sum
        self.totals.log10_sum += log10_sum
        self.totals.delta1_hits += delta1_hits
        self.totals.delta2_hits += delta2_hits
        self.totals.delta3_hits += delta3_hits
        self.totals.silog_log_sum += silog_log_sum
        self.totals.silog_log_sq_sum += silog_log_sq_sum

    def merge(self, other: "DepthMetricAccumulator") -> None:
        self.totals.add(other.totals)

    def finalize(self) -> Dict[str, float]:
        count = self.totals.valid_count
        if count == 0:
            return {
                "abs_rel": float("nan"),
                "sq_rel": float("nan"),
                "rmse": float("nan"),
                "rmse_log": float("nan"),
                "log10": float("nan"),
                "silog": float("nan"),
                "delta1": float("nan"),
                "delta2": float("nan"),
                "delta3": float("nan"),
                "valid_count": 0,
            }

        mean_log_diff = self.totals.silog_log_sum / count
        mean_log_diff_sq = self.totals.silog_log_sq_sum / count
        silog = math.sqrt(max(mean_log_diff_sq - mean_log_diff * mean_log_diff, 0.0)) * 100.0
        return {
            "abs_rel": self.totals.abs_rel_sum / count,
            "sq_rel": self.totals.sq_rel_sum / count,
            "rmse": math.sqrt(self.totals.rmse_sq_sum / count),
            "rmse_log": math.sqrt(self.totals.rmse_log_sq_sum / count),
            "log10": self.totals.log10_sum / count,
            "silog": silog,
            "delta1": self.totals.delta1_hits / count,
            "delta2": self.totals.delta2_hits / count,
            "delta3": self.totals.delta3_hits / count,
            "valid_count": count,
        }


def compute_metrics(predictions, targets=None, masks=None, align_mode: str = "none"):
    """Convenience wrapper for either a single pair or a batch of pairs."""
    accumulator = DepthMetricAccumulator(align_mode=align_mode)

    if targets is None and isinstance(predictions, Iterable):
        for item in predictions:
            if len(item) == 3:
                pred_depth, gt_depth, mask = item
            elif len(item) == 2:
                pred_depth, gt_depth = item
                mask = None
            else:
                raise ValueError("Each item must contain 2 or 3 elements")
            accumulator.update(pred_depth, gt_depth, mask)
        return accumulator.finalize()

    if targets is None:
        raise ValueError("targets must be provided when predictions is not an iterable of samples")

    accumulator.update(predictions, targets, masks)
    return accumulator.finalize()
