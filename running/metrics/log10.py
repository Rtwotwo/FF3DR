from __future__ import annotations

import numpy as np

from .common import EPS, build_valid_mask


def accumulate_log10(pred_depth, gt_depth, mask=None):
    valid = build_valid_mask(pred_depth, gt_depth, mask=mask)
    valid_count = int(valid.sum())
    if valid_count == 0:
        return 0.0, 0
    pred_valid = np.log10(np.clip(np.asarray(pred_depth)[valid], EPS, None))
    gt_valid = np.log10(np.clip(np.asarray(gt_depth)[valid], EPS, None))
    value = np.abs(pred_valid - gt_valid)
    return float(value.sum()), valid_count
