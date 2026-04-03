from __future__ import annotations

import numpy as np

from .common import build_valid_mask


def accumulate_rmse(pred_depth, gt_depth, mask=None):
    valid = build_valid_mask(pred_depth, gt_depth, mask=mask)
    valid_count = int(valid.sum())
    if valid_count == 0:
        return 0.0, 0
    pred_valid = np.asarray(pred_depth)[valid]
    gt_valid = np.asarray(gt_depth)[valid]
    value = np.square(pred_valid - gt_valid)
    return float(value.sum()), valid_count
