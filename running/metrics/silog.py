from __future__ import annotations

import numpy as np

from .common import EPS, build_valid_mask


def accumulate_silog_terms(pred_depth, gt_depth, mask=None):
    valid = build_valid_mask(pred_depth, gt_depth, mask=mask)
    valid_count = int(valid.sum())
    if valid_count == 0:
        return 0.0, 0.0, 0
    pred_valid = np.log(np.clip(np.asarray(pred_depth)[valid], EPS, None))
    gt_valid = np.log(np.clip(np.asarray(gt_depth)[valid], EPS, None))
    diff = pred_valid - gt_valid
    return float(diff.sum()), float(np.square(diff).sum()), valid_count
