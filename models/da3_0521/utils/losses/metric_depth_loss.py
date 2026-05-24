from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetricDepthLossV3(nn.Module):
    def __init__(
        self,
        si_weight: float = 1.0,
        logl1_weight: float = 10.0,
        l1_weight: float = 1.0,
        absrel_weight: float = 0.5,
        gradient_weight: float = 0.5,
        range_weight: float = 0.1,
        scale_reg_weight: float = 0.01,
        shift_reg_weight: float = 0.01,
        depth_range: Tuple[float, float] = (1.0, 2000.0),
        depth_norm: float = 600.0,
        charbonnier_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.si_weight = si_weight
        self.logl1_weight = logl1_weight
        self.l1_weight = l1_weight
        self.absrel_weight = absrel_weight
        self.gradient_weight = gradient_weight
        self.range_weight = range_weight
        self.scale_reg_weight = scale_reg_weight
        self.shift_reg_weight = shift_reg_weight
        self.depth_range = depth_range
        self.depth_norm = depth_norm
        self.charbonnier_eps = charbonnier_eps

    def forward(
        self,
        pred_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        log_scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if valid_mask is None:
            valid_mask = (gt_depth > self.depth_range[0]) & (gt_depth < self.depth_range[1])
        else:
            valid_mask = (valid_mask > 0.5) & (gt_depth > self.depth_range[0]) & (gt_depth < self.depth_range[1])

        if valid_mask.sum() < 10:
            zero = torch.tensor(0.0, device=pred_depth.device)
            return {
                "loss": zero,
                "si_loss": zero.clone(),
                "logl1_loss": zero.clone(),
                "l1_loss": zero.clone(),
                "absrel_loss": zero.clone(),
                "gradient_loss": zero.clone(),
                "range_loss": zero.clone(),
                "scale_reg": zero.clone(),
                "shift_reg": zero.clone(),
            }

        si_loss = self._scale_invariant_log_loss(pred_depth, gt_depth, valid_mask)
        logl1_loss = self._log_depth_l1_loss(pred_depth, gt_depth, valid_mask)
        l1_loss = self._charbonnier_loss(pred_depth, gt_depth, valid_mask)
        absrel_loss = self._abs_rel_loss(pred_depth, gt_depth, valid_mask)
        gradient_loss = self._multi_scale_gradient_loss(pred_depth, gt_depth, valid_mask)
        range_loss = self._depth_range_loss(pred_depth, valid_mask)

        scale_reg = torch.tensor(0.0, device=pred_depth.device)
        shift_reg = torch.tensor(0.0, device=pred_depth.device)
        if log_scale is not None:
            scale_reg = (log_scale ** 2).mean()
        if shift is not None:
            shift_reg = (shift / max(self.depth_norm, 1.0)).pow(2).mean()

        total_loss = (
            self.si_weight * si_loss
            + self.logl1_weight * logl1_loss
            + self.l1_weight * l1_loss
            + self.absrel_weight * absrel_loss
            + self.gradient_weight * gradient_loss
            + self.range_weight * range_loss
            + self.scale_reg_weight * scale_reg
            + self.shift_reg_weight * shift_reg
        )

        return {
            "loss": total_loss,
            "si_loss": si_loss.detach(),
            "logl1_loss": logl1_loss.detach(),
            "l1_loss": l1_loss.detach(),
            "absrel_loss": absrel_loss.detach(),
            "gradient_loss": gradient_loss.detach(),
            "range_loss": range_loss.detach(),
            "scale_reg": scale_reg.detach(),
            "shift_reg": shift_reg.detach(),
        }

    def _scale_invariant_log_loss(self, pred, gt, valid_mask):
        log_pred = torch.log(pred[valid_mask].clamp(min=1e-6))
        log_gt = torch.log(gt[valid_mask].clamp(min=1e-6))
        diff = log_pred - log_gt
        return (diff ** 2).mean() - 0.5 * (diff.mean()) ** 2

    def _log_depth_l1_loss(self, pred, gt, valid_mask):
        log_pred = torch.log1p(pred[valid_mask].clamp(min=0.0) / max(self.depth_norm, 1.0))
        log_gt = torch.log1p(gt[valid_mask].clamp(min=0.0) / max(self.depth_norm, 1.0))
        return F.l1_loss(log_pred, log_gt)

    def _charbonnier_loss(self, pred, gt, valid_mask):
        diff = pred[valid_mask] - gt[valid_mask]
        return torch.sqrt(diff * diff + self.charbonnier_eps ** 2).mean()

    def _abs_rel_loss(self, pred, gt, valid_mask):
        g = gt[valid_mask].clamp(min=1e-6)
        return torch.mean(torch.abs(pred[valid_mask] - g) / g)

    def _multi_scale_gradient_loss(self, pred, gt, valid_mask, scales=(1, 2, 4)):
        if pred.ndim == 2:
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)
            valid_mask = valid_mask.unsqueeze(0)
        if pred.ndim != 3 or gt.ndim != 3:
            return torch.tensor(0.0, device=pred.device)

        total = torch.tensor(0.0, device=pred.device)
        count = 0
        for b in range(pred.shape[0]):
            p_b = pred[b]
            g_b = gt[b]
            m_b = valid_mask[b]
            for s in scales:
                if s > 1:
                    p = F.avg_pool2d(p_b.unsqueeze(0).unsqueeze(0), s).squeeze(0).squeeze(0)
                    g = F.avg_pool2d(g_b.unsqueeze(0).unsqueeze(0), s).squeeze(0).squeeze(0)
                    m = (
                        F.avg_pool2d(m_b.float().unsqueeze(0).unsqueeze(0), s)
                        .squeeze(0)
                        .squeeze(0)
                        > 0.5
                    )
                else:
                    p, g, m = p_b, g_b, m_b

                p_dx = p[:, 1:] - p[:, :-1]
                p_dy = p[1:, :] - p[:-1, :]
                g_dx = g[:, 1:] - g[:, :-1]
                g_dy = g[1:, :] - g[:-1, :]
                m_dx = m[:, 1:] & m[:, :-1]
                m_dy = m[1:, :] & m[:-1, :]

                if m_dx.sum() > 0:
                    total = total + F.l1_loss(p_dx[m_dx], g_dx[m_dx])
                    count += 1
                if m_dy.sum() > 0:
                    total = total + F.l1_loss(p_dy[m_dy], g_dy[m_dy])
                    count += 1

        return total / max(count, 1)

    def _depth_range_loss(self, pred, valid_mask):
        lo = self.depth_range[0]
        hi = self.depth_range[1]
        below = F.relu(lo - pred[valid_mask])
        above = F.relu(pred[valid_mask] - hi)
        return (below.mean() + above.mean())
