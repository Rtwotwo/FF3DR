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


class MetricDepthLossV4(nn.Module):
    def __init__(
        self,
        relative_si_weight: float = 0.5,
        metric_si_weight: float = 1.0,
        logl1_weight: float = 6.0,
        l1_weight: float = 0.5,
        absrel_weight: float = 0.25,
        gradient_weight: float = 0.35,
        range_weight: float = 0.05,
        confidence_weight: float = 0.2,
        sky_weight: float = 0.1,
        scale_reg_weight: float = 0.01,
        shift_reg_weight: float = 0.01,
        depth_range: Tuple[float, float] = (1.0, 2000.0),
        depth_norm: float = 600.0,
        confidence_tau: float = 120.0,
        sky_threshold: float = 0.3,
        far_depth_start: float = 350.0,
        far_depth_boost: float = 0.35,
        far_depth_transition: float = 80.0,
        charbonnier_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.relative_si_weight = relative_si_weight
        self.metric_si_weight = metric_si_weight
        self.logl1_weight = logl1_weight
        self.l1_weight = l1_weight
        self.absrel_weight = absrel_weight
        self.gradient_weight = gradient_weight
        self.range_weight = range_weight
        self.confidence_weight = confidence_weight
        self.sky_weight = sky_weight
        self.scale_reg_weight = scale_reg_weight
        self.shift_reg_weight = shift_reg_weight
        self.depth_range = depth_range
        self.depth_norm = depth_norm
        self.confidence_tau = confidence_tau
        self.sky_threshold = sky_threshold
        self.far_depth_start = far_depth_start
        self.far_depth_boost = far_depth_boost
        self.far_depth_transition = far_depth_transition
        self.charbonnier_eps = charbonnier_eps

    def forward(
        self,
        pred_depth: torch.Tensor,
        pred_metric: torch.Tensor,
        gt_depth: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        pred_conf: Optional[torch.Tensor] = None,
        pred_sky: Optional[torch.Tensor] = None,
        log_scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pred_depth = self._squeeze_depth_like(pred_depth)
        pred_metric = self._squeeze_depth_like(pred_metric)
        gt_depth = self._squeeze_depth_like(gt_depth)
        valid_mask = self._squeeze_mask_like(valid_mask) if valid_mask is not None else None
        pred_conf = self._squeeze_depth_like(pred_conf) if pred_conf is not None else None
        pred_sky = self._squeeze_depth_like(pred_sky) if pred_sky is not None else None

        if valid_mask is None:
            valid_mask = (gt_depth > self.depth_range[0]) & (gt_depth < self.depth_range[1])
        else:
            valid_mask = (valid_mask > 0.5) & (gt_depth > self.depth_range[0]) & (gt_depth < self.depth_range[1])

        if valid_mask.sum() < 10:
            zero = torch.tensor(0.0, device=pred_metric.device)
            return {
                "loss": zero,
                "relative_si_loss": zero.clone(),
                "metric_si_loss": zero.clone(),
                "logl1_loss": zero.clone(),
                "l1_loss": zero.clone(),
                "absrel_loss": zero.clone(),
                "gradient_loss": zero.clone(),
                "range_loss": zero.clone(),
                "confidence_loss": zero.clone(),
                "sky_loss": zero.clone(),
                "scale_reg": zero.clone(),
                "shift_reg": zero.clone(),
            }

        if pred_sky is not None:
            non_sky_mask = pred_sky < self.sky_threshold
            sky_prob = 1.0 - torch.exp(-pred_sky.clamp(min=0.0))
            sky_target = (~valid_mask).float()
            sky_weight_map = torch.where(valid_mask, torch.ones_like(gt_depth), torch.full_like(gt_depth, self.sky_weight))
            sky_loss = self._weighted_l1(sky_prob, sky_target, sky_weight_map)
        else:
            non_sky_mask = torch.ones_like(valid_mask, dtype=torch.bool)
            sky_loss = torch.tensor(0.0, device=pred_metric.device)

        metric_mask = valid_mask & non_sky_mask
        if metric_mask.sum() < 10:
            metric_mask = valid_mask

        weight_map = torch.where(non_sky_mask, torch.ones_like(gt_depth), torch.full_like(gt_depth, 0.25))
        depth_weight = self._depth_reweight(gt_depth)
        weight_map = weight_map * depth_weight

        confidence_loss = torch.tensor(0.0, device=pred_metric.device)
        if pred_conf is not None:
            conf_score = torch.sigmoid(torch.log1p(pred_conf.clamp(min=1e-6)))
            target_conf = torch.exp(-torch.abs(pred_metric - gt_depth) / max(self.confidence_tau, 1.0)).detach()
            confidence_loss = self._weighted_smooth_l1(conf_score, target_conf, valid_mask.float())
            weight_map = weight_map * (0.5 + 0.5 * target_conf.detach())

        relative_si_loss = self._weighted_scale_invariant_log_loss(pred_depth, gt_depth, metric_mask)
        metric_si_loss = self._weighted_scale_invariant_log_loss(pred_metric, gt_depth, metric_mask)
        logl1_loss = self._weighted_log_depth_l1_loss(pred_metric, gt_depth, metric_mask, weight_map)
        l1_loss = self._weighted_charbonnier_loss(pred_metric, gt_depth, metric_mask, weight_map)
        absrel_loss = self._weighted_abs_rel_loss(pred_metric, gt_depth, metric_mask, weight_map)
        gradient_loss = self._weighted_multi_scale_gradient_loss(pred_metric, gt_depth, metric_mask, weight_map)
        range_loss = self._depth_range_loss(pred_metric, metric_mask)

        scale_reg = torch.tensor(0.0, device=pred_metric.device)
        shift_reg = torch.tensor(0.0, device=pred_metric.device)
        if log_scale is not None:
            scale_reg = (log_scale ** 2).mean()
        if shift is not None:
            shift_reg = (shift / max(self.depth_norm, 1.0)).pow(2).mean()

        total_loss = (
            self.relative_si_weight * relative_si_loss
            + self.metric_si_weight * metric_si_loss
            + self.logl1_weight * logl1_loss
            + self.l1_weight * l1_loss
            + self.absrel_weight * absrel_loss
            + self.gradient_weight * gradient_loss
            + self.range_weight * range_loss
            + self.confidence_weight * confidence_loss
            + self.sky_weight * sky_loss
            + self.scale_reg_weight * scale_reg
            + self.shift_reg_weight * shift_reg
        )

        return {
            "loss": total_loss,
            "relative_si_loss": relative_si_loss.detach(),
            "metric_si_loss": metric_si_loss.detach(),
            "logl1_loss": logl1_loss.detach(),
            "l1_loss": l1_loss.detach(),
            "absrel_loss": absrel_loss.detach(),
            "gradient_loss": gradient_loss.detach(),
            "range_loss": range_loss.detach(),
            "confidence_loss": confidence_loss.detach(),
            "sky_loss": sky_loss.detach(),
            "scale_reg": scale_reg.detach(),
            "shift_reg": shift_reg.detach(),
        }

    def _reduce_with_mask(self, value: torch.Tensor, weight_map: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        value = value.float()
        valid_mask = valid_mask.to(dtype=torch.bool)
        try:
            value, weight_map, valid_mask = torch.broadcast_tensors(value, weight_map.float(), valid_mask)
        except RuntimeError:
            zero = torch.tensor(0.0, device=value.device)
            return zero

        value = value.reshape(-1)
        weight_map = weight_map.reshape(-1)
        valid_mask = valid_mask.reshape(-1)

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=value.device)

        weight = weight_map[valid_mask]
        if weight.numel() == 0:
            return torch.tensor(0.0, device=value.device)
        denom = weight.sum().clamp_min(1e-6)
        return (value[valid_mask] * weight).sum() / denom

    def _squeeze_depth_like(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if tensor.ndim == 4 and tensor.shape[1] == 1:
            return tensor[:, 0]
        if tensor.ndim == 4 and tensor.shape[1] > 1:
            return tensor[:, 0]
        return tensor

    def _squeeze_mask_like(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if tensor.ndim == 4 and tensor.shape[1] == 1:
            return tensor[:, 0].bool()
        if tensor.ndim == 4 and tensor.shape[1] > 1:
            return tensor[:, 0].bool()
        return tensor.bool()

    def _depth_reweight(self, gt_depth: torch.Tensor) -> torch.Tensor:
        depth = self._squeeze_depth_like(gt_depth).float().clamp(min=0.0)
        transition = max(self.far_depth_transition, 1.0)
        far_ratio = torch.sigmoid((depth - self.far_depth_start) / transition)
        return 1.0 + self.far_depth_boost * far_ratio

    def _weighted_l1(self, pred, gt, weight_map):
        pred = self._squeeze_depth_like(pred)
        gt = self._squeeze_depth_like(gt)
        weight_map = self._squeeze_depth_like(weight_map)
        loss = torch.abs(pred - gt)
        return self._reduce_with_mask(loss, weight_map, torch.ones_like(loss, dtype=torch.bool))

    def _weighted_smooth_l1(self, pred, gt, weight_map):
        pred = self._squeeze_depth_like(pred)
        gt = self._squeeze_depth_like(gt)
        weight_map = self._squeeze_depth_like(weight_map)
        loss = F.smooth_l1_loss(pred, gt, reduction="none")
        return self._reduce_with_mask(loss, weight_map, torch.ones_like(loss, dtype=torch.bool))

    def _weighted_scale_invariant_log_loss(self, pred, gt, valid_mask):
        pred = self._squeeze_depth_like(pred)
        gt = self._squeeze_depth_like(gt)
        valid_mask = self._squeeze_mask_like(valid_mask)
        log_pred = torch.log(pred[valid_mask].clamp(min=1e-6))
        log_gt = torch.log(gt[valid_mask].clamp(min=1e-6))
        diff = log_pred - log_gt
        return (diff ** 2).mean() - 0.5 * (diff.mean()) ** 2

    def _weighted_log_depth_l1_loss(self, pred, gt, valid_mask, weight_map=None):
        pred = self._squeeze_depth_like(pred)
        gt = self._squeeze_depth_like(gt)
        valid_mask = self._squeeze_mask_like(valid_mask)
        if weight_map is None:
            weight_map = torch.ones_like(gt)
        weight_map = self._squeeze_depth_like(weight_map)
        log_pred = torch.log1p(pred[valid_mask].clamp(min=0.0) / max(self.depth_norm, 1.0))
        log_gt = torch.log1p(gt[valid_mask].clamp(min=0.0) / max(self.depth_norm, 1.0))
        log_diff = torch.abs(log_pred - log_gt)
        return self._reduce_with_mask(log_diff, weight_map[valid_mask], torch.ones_like(log_diff, dtype=torch.bool))

    def _weighted_charbonnier_loss(self, pred, gt, valid_mask, weight_map=None):
        pred = self._squeeze_depth_like(pred)
        gt = self._squeeze_depth_like(gt)
        valid_mask = self._squeeze_mask_like(valid_mask)
        if weight_map is None:
            weight_map = torch.ones_like(gt)
        weight_map = self._squeeze_depth_like(weight_map)
        diff = pred[valid_mask] - gt[valid_mask]
        loss = torch.sqrt(diff * diff + self.charbonnier_eps ** 2)
        return self._reduce_with_mask(loss, weight_map[valid_mask], torch.ones_like(loss, dtype=torch.bool))

    def _weighted_abs_rel_loss(self, pred, gt, valid_mask, weight_map=None):
        pred = self._squeeze_depth_like(pred)
        gt = self._squeeze_depth_like(gt)
        valid_mask = self._squeeze_mask_like(valid_mask)
        if weight_map is None:
            weight_map = torch.ones_like(gt)
        weight_map = self._squeeze_depth_like(weight_map)
        g = gt[valid_mask].clamp(min=1e-6)
        loss = torch.abs(pred[valid_mask] - g) / g
        return self._reduce_with_mask(loss, weight_map[valid_mask], torch.ones_like(loss, dtype=torch.bool))

    def _weighted_multi_scale_gradient_loss(self, pred, gt, valid_mask, weight_map, scales=(1, 2, 4)):
        if pred.ndim == 2:
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)
            valid_mask = valid_mask.unsqueeze(0)
            weight_map = weight_map.unsqueeze(0)
        if pred.ndim != 3 or gt.ndim != 3:
            return torch.tensor(0.0, device=pred.device)

        total = torch.tensor(0.0, device=pred.device)
        count = 0
        for b in range(pred.shape[0]):
            p_b = pred[b]
            g_b = gt[b]
            m_b = valid_mask[b]
            w_b = weight_map[b]
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
                    w = F.avg_pool2d(w_b.unsqueeze(0).unsqueeze(0), s).squeeze(0).squeeze(0)
                else:
                    p, g, m, w = p_b, g_b, m_b, w_b

                p_dx = p[:, 1:] - p[:, :-1]
                p_dy = p[1:, :] - p[:-1, :]
                g_dx = g[:, 1:] - g[:, :-1]
                g_dy = g[1:, :] - g[:-1, :]
                m_dx = m[:, 1:] & m[:, :-1]
                m_dy = m[1:, :] & m[:-1, :]

                if m_dx.sum() > 0:
                    total = total + self._reduce_with_mask(torch.abs(p_dx - g_dx), w[:, 1:] * w[:, :-1], m_dx)
                    count += 1
                if m_dy.sum() > 0:
                    total = total + self._reduce_with_mask(torch.abs(p_dy - g_dy), w[1:, :] * w[:-1, :], m_dy)
                    count += 1

        return total / max(count, 1)

    def _depth_range_loss(self, pred, valid_mask):
        pred = self._squeeze_depth_like(pred)
        valid_mask = self._squeeze_mask_like(valid_mask)
        lo = self.depth_range[0]
        hi = self.depth_range[1]
        below = F.relu(lo - pred[valid_mask])
        above = F.relu(pred[valid_mask] - hi)
        return (below.mean() + above.mean())


class DepthOnlyLoss(nn.Module):
    def __init__(
        self,
        si_weight: float = 0.0,
        logl1_weight: float = 0.1,
        l1_weight: float = 1.0,
        gradient_weight: float = 0.1,
        range_weight: float = 0.0,
        depth_range: Tuple[float, float] = (1.0, 2000.0),
        charbonnier_eps: float = 1e-3,
        median_align: bool = False,
    ) -> None:
        super().__init__()
        self.si_weight = si_weight
        self.logl1_weight = logl1_weight
        self.l1_weight = l1_weight
        self.gradient_weight = gradient_weight
        self.range_weight = range_weight
        self.depth_range = depth_range
        self.charbonnier_eps = charbonnier_eps
        self.median_align = median_align

    def forward(
        self,
        pred_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pred_depth = self._squeeze_depth_like(pred_depth)
        gt_depth = self._squeeze_depth_like(gt_depth)
        valid_mask = self._squeeze_mask_like(valid_mask) if valid_mask is not None else None

        if pred_depth.ndim == 2:
            pred_depth = pred_depth.unsqueeze(0)
            gt_depth = gt_depth.unsqueeze(0)
            if valid_mask is not None:
                valid_mask = valid_mask.unsqueeze(0)
        if pred_depth.ndim != 3 or gt_depth.ndim != 3:
            zero = torch.tensor(0.0, device=pred_depth.device)
            return {
                "loss": zero,
                "si_loss": zero.clone(),
                "logl1_loss": zero.clone(),
                "l1_loss": zero.clone(),
                "gradient_loss": zero.clone(),
                "range_loss": zero.clone(),
                "median_scale": zero.clone(),
            }

        total_loss = torch.tensor(0.0, device=pred_depth.device)
        total_si_loss = torch.tensor(0.0, device=pred_depth.device)
        total_logl1_loss = torch.tensor(0.0, device=pred_depth.device)
        total_l1_loss = torch.tensor(0.0, device=pred_depth.device)
        total_gradient_loss = torch.tensor(0.0, device=pred_depth.device)
        total_range_loss = torch.tensor(0.0, device=pred_depth.device)
        total_median_scale = torch.tensor(0.0, device=pred_depth.device)
        valid_batches = 0

        for batch_idx in range(pred_depth.shape[0]):
            pred = pred_depth[batch_idx]
            gt = gt_depth[batch_idx]
            if valid_mask is None:
                mask = (gt > self.depth_range[0]) & (gt < self.depth_range[1])
            else:
                mask = valid_mask[batch_idx].bool() & (gt > self.depth_range[0]) & (gt < self.depth_range[1])
            mask = mask & torch.isfinite(pred) & torch.isfinite(gt)
            if mask.sum() < 10:
                continue

            pred_valid = pred[mask]
            gt_valid = gt[mask]
            if self.median_align:
                scale = (gt_valid.median() / pred_valid.median().clamp_min(1e-6)).detach()
                pred = pred * scale
                total_median_scale = total_median_scale + scale.detach()
            else:
                total_median_scale = total_median_scale + torch.tensor(1.0, device=pred_depth.device)

            si_loss = self._scale_invariant_log_loss(pred, gt, mask)
            logl1_loss = self._log_depth_l1_loss(pred, gt, mask)
            l1_loss = self._charbonnier_loss(pred, gt, mask)
            gradient_loss = self._multi_scale_gradient_loss(pred, gt, mask)
            range_loss = self._depth_range_loss(pred, mask)

            total_si_loss = total_si_loss + si_loss
            total_logl1_loss = total_logl1_loss + logl1_loss
            total_l1_loss = total_l1_loss + l1_loss
            total_gradient_loss = total_gradient_loss + gradient_loss
            total_range_loss = total_range_loss + range_loss
            total_loss = total_loss + (
                self.si_weight * si_loss
                + self.logl1_weight * logl1_loss
                + self.l1_weight * l1_loss
                + self.gradient_weight * gradient_loss
                + self.range_weight * range_loss
            )
            valid_batches += 1

        if valid_batches == 0:
            zero = torch.tensor(0.0, device=pred_depth.device)
            return {
                "loss": zero,
                "si_loss": zero.clone(),
                "logl1_loss": zero.clone(),
                "l1_loss": zero.clone(),
                "gradient_loss": zero.clone(),
                "range_loss": zero.clone(),
                "median_scale": zero.clone(),
            }

        denom = float(valid_batches)
        return {
            "loss": total_loss / denom,
            "si_loss": total_si_loss / denom,
            "logl1_loss": total_logl1_loss / denom,
            "l1_loss": total_l1_loss / denom,
            "gradient_loss": total_gradient_loss / denom,
            "range_loss": total_range_loss / denom,
            "median_scale": total_median_scale / denom,
        }

    def _squeeze_depth_like(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if tensor.ndim == 4 and tensor.shape[1] == 1:
            return tensor[:, 0]
        if tensor.ndim == 4 and tensor.shape[1] > 1:
            return tensor[:, 0]
        return tensor

    def _squeeze_mask_like(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if tensor.ndim == 4 and tensor.shape[1] == 1:
            return tensor[:, 0].bool()
        if tensor.ndim == 4 and tensor.shape[1] > 1:
            return tensor[:, 0].bool()
        return tensor.bool()

    def _scale_invariant_log_loss(self, pred, gt, valid_mask):
        log_pred = torch.log(pred[valid_mask].clamp(min=1e-6))
        log_gt = torch.log(gt[valid_mask].clamp(min=1e-6))
        diff = log_pred - log_gt
        return (diff ** 2).mean() - 0.5 * (diff.mean()) ** 2

    def _log_depth_l1_loss(self, pred, gt, valid_mask):
        log_pred = torch.log1p(pred[valid_mask].clamp(min=0.0) / max(self.depth_range[1], 1.0))
        log_gt = torch.log1p(gt[valid_mask].clamp(min=0.0) / max(self.depth_range[1], 1.0))
        return F.l1_loss(log_pred, log_gt)

    def _charbonnier_loss(self, pred, gt, valid_mask):
        diff = pred[valid_mask] - gt[valid_mask]
        return torch.sqrt(diff * diff + self.charbonnier_eps ** 2).mean()

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
