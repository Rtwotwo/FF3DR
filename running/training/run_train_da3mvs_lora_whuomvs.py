from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR
while _REPO_ROOT != _REPO_ROOT.parent and not (_REPO_ROOT / "models").exists():
    _REPO_ROOT = _REPO_ROOT.parent
for p in (_REPO_ROOT, _REPO_ROOT / "models", _SCRIPT_DIR.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from running.data.whuomvs_depth_dataset import WHUOMVSDepthDataset
from running.training.da3_metric_loss import MetricDepthLossV4
from models.adamvs.adamvs import AdaMVSNet

logger = logging.getLogger(__name__)


class MetricScaleShift(nn.Module):
    def __init__(self, init_scale: float = 1.0, init_shift: float = 0.0):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(np.log(max(init_scale, 1e-6)), dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor(init_shift, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.exp(self.log_scale) + self.shift

    @property
    def scale(self) -> float:
        return torch.exp(self.log_scale).item()

    @property
    def shift_val(self) -> float:
        return self.shift.item()


class MetricDepthLossV2(nn.Module):
    def __init__(
        self,
        si_weight: float = 1.0,
        logl1_weight: float = 10.0,
        l1_weight: float = 1.0,
        gradient_weight: float = 0.5,
        range_weight: float = 0.1,
        depth_range: Tuple[float, float] = (1.0, 2000.0),
    ):
        super().__init__()
        self.si_weight = si_weight
        self.logl1_weight = logl1_weight
        self.l1_weight = l1_weight
        self.gradient_weight = gradient_weight
        self.range_weight = range_weight
        self.depth_range = depth_range

    def forward(
        self,
        pred_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if valid_mask is None:
            valid_mask = (gt_depth > self.depth_range[0]) & (gt_depth < self.depth_range[1])
        else:
            valid_mask = valid_mask.bool() & (gt_depth > self.depth_range[0]) & (gt_depth < self.depth_range[1])

        if valid_mask.sum() < 10:
            zero = torch.tensor(0.0, device=pred_depth.device)
            return {
                "loss": zero,
                "si_loss": zero.clone(),
                "logl1_loss": zero.clone(),
                "l1_loss": zero.clone(),
                "gradient_loss": zero.clone(),
                "range_loss": zero.clone(),
            }

        si_loss = self._scale_invariant_log_loss(pred_depth, gt_depth, valid_mask)
        logl1_loss = self._log_depth_l1_loss(pred_depth, gt_depth, valid_mask)
        l1_loss = F.l1_loss(pred_depth[valid_mask], gt_depth[valid_mask])
        gradient_loss = self._multi_scale_gradient_loss(pred_depth, gt_depth, valid_mask)
        range_loss = self._depth_range_loss(pred_depth, valid_mask)

        total_loss = (
            self.si_weight * si_loss
            + self.logl1_weight * logl1_loss
            + self.l1_weight * l1_loss
            + self.gradient_weight * gradient_loss
            + self.range_weight * range_loss
        )

        return {
            "loss": total_loss,
            "si_loss": si_loss.detach(),
            "logl1_loss": logl1_loss.detach(),
            "l1_loss": l1_loss.detach(),
            "gradient_loss": gradient_loss.detach(),
            "range_loss": range_loss.detach(),
        }

    def _scale_invariant_log_loss(self, pred, gt, valid_mask):
        log_pred = torch.log(pred[valid_mask].clamp(min=1e-6))
        log_gt = torch.log(gt[valid_mask].clamp(min=1e-6))
        diff = log_pred - log_gt
        return (diff ** 2).mean() - 0.5 * (diff.mean()) ** 2

    def _log_depth_l1_loss(self, pred, gt, valid_mask):
        log_pred = torch.log(pred[valid_mask].clamp(min=1e-6))
        log_gt = torch.log(gt[valid_mask].clamp(min=1e-6))
        return F.l1_loss(log_pred, log_gt)

    def _multi_scale_gradient_loss(self, pred, gt, valid_mask, scales=(1, 2, 4)):
        if pred.ndim != 2 or gt.ndim != 2:
            return torch.tensor(0.0, device=pred.device)
        total = torch.tensor(0.0, device=pred.device)
        count = 0
        for s in scales:
            if s > 1:
                p = F.avg_pool2d(pred.unsqueeze(0).unsqueeze(0), s).squeeze(0).squeeze(0)
                g = F.avg_pool2d(gt.unsqueeze(0).unsqueeze(0), s).squeeze(0).squeeze(0)
                m = F.avg_pool2d(valid_mask.float().unsqueeze(0).unsqueeze(0), s).squeeze(0).squeeze(0) > 0.5
            else:
                p, g, m = pred, gt, valid_mask

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


def _build_adamvs_proj_matrices(extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> Dict[str, torch.Tensor]:
    if extrinsics.ndim == 3:
        extrinsics = extrinsics.unsqueeze(1)
    if intrinsics.ndim == 3:
        intrinsics = intrinsics.unsqueeze(1)
    if extrinsics.ndim != 4 or intrinsics.ndim != 4:
        raise ValueError(f"Expected batched extrinsics/intrinsics, got {extrinsics.shape} / {intrinsics.shape}")
    if extrinsics.shape[:2] != intrinsics.shape[:2]:
        raise ValueError(f"Extrinsics/intrinsics batch mismatch: {extrinsics.shape} vs {intrinsics.shape}")

    proj_matrices = torch.matmul(intrinsics, extrinsics[:, :, :3, :])
    proj_matrices = F.pad(proj_matrices, (0, 0, 0, 1), value=0.0)
    proj_matrices[:, :, 3, 3] = 1.0

    stage1 = proj_matrices.clone()
    stage1[:, :, :2, :] = proj_matrices[:, :, :2, :] / 4.0
    stage2 = proj_matrices.clone()
    stage2[:, :, :2, :] = proj_matrices[:, :, :2, :] / 2.0

    return {"stage1": stage1, "stage2": stage2, "stage3": proj_matrices}


def _build_adamvs_depth_values(
    depth_gt: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    num_depth: int,
    fallback_min: float = 1.0,
    fallback_max: float = 2000.0,
) -> torch.Tensor:
    if depth_gt.ndim != 3:
        raise ValueError(f"Expected depth_gt shape [B,H,W], got {depth_gt.shape}")
    if valid_mask is None:
        valid_mask = torch.isfinite(depth_gt) & (depth_gt > 0)
    else:
        valid_mask = valid_mask.bool() & torch.isfinite(depth_gt) & (depth_gt > 0)

    values = []
    for b in range(depth_gt.shape[0]):
        valid = depth_gt[b][valid_mask[b]]
        if valid.numel() < 10:
            depth_min = fallback_min
            depth_max = fallback_max
        else:
            depth_min = float(valid.min().item())
            depth_max = float(valid.max().item())
            if not math.isfinite(depth_min):
                depth_min = fallback_min
            if not math.isfinite(depth_max):
                depth_max = fallback_max
            if depth_max <= depth_min + 1e-3:
                depth_max = depth_min + 1.0
        depth_interval = max((depth_max - depth_min) / max(num_depth - 1, 1), 1e-3)
        values.append([depth_min, depth_max, depth_interval])
    return torch.tensor(values, device=depth_gt.device, dtype=depth_gt.dtype)


class AdaMVSPreDepthFeatureExtractor(nn.Module):
    """Frozen Ada-MVS depth-branch feature extractor.

    This wrapper runs the full Ada-MVS forward pass and exposes the final
    stage pre-depth fused feature map from the MVS branch, not the plain
    image encoder output.
    """

    def __init__(self, ckpt_path: Optional[str], trainable: bool = False):
        super().__init__()
        self.model = AdaMVSNet(ndepths=[48, 32, 8], depth_intervals_ratio=[4, 2, 1], share_cr=False, cr_base_chs=[8, 8, 8])
        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model_state = ckpt.get("model", ckpt)
            missing, unexpected = self.model.load_state_dict(model_state, strict=False)
            logger.info(
                "Loaded Ada-MVS pre-depth model from %s (missing=%d, unexpected=%d)",
                ckpt_path,
                len(missing),
                len(unexpected),
            )
        else:
            logger.warning("Ada-MVS checkpoint not found, using random depth-branch model: %s", ckpt_path)

        if trainable:
            logger.warning("Ada-MVS branch is frozen in this recipe; ignoring trainable=True")

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(
        self,
        images_bnvchw: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        depth_gt: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        proj_matrices = _build_adamvs_proj_matrices(extrinsics, intrinsics)
        depth_values = _build_adamvs_depth_values(depth_gt, valid_mask, num_depth=48)
        with torch.no_grad():
            outputs = self.model(images_bnvchw, proj_matrices, depth_values, return_pre_depth_feature=True)
        feature_map = {
            "stage2": outputs.get("pre_depth_feature_stage2", None),
            "stage3": outputs.get("pre_depth_feature_stage3", None),
        }
        if feature_map["stage3"] is None:
            raise RuntimeError("Ada-MVS forward did not return stage3 pre-depth feature")
        if feature_map["stage2"] is None:
            feature_map["stage2"] = feature_map["stage3"]
        return feature_map


class DA3MVSFusionHead(nn.Module):
    def __init__(self, da3_in_dim: int, ada_in_dim: int, fusion_dim: int = 128):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.da3_proj = nn.Conv2d(da3_in_dim, fusion_dim, kernel_size=1, stride=1, padding=0)
        self.ada_proj = nn.Conv2d(ada_in_dim, fusion_dim, kernel_size=1, stride=1, padding=0)
        self.fuse = nn.Sequential(
            nn.Conv2d(fusion_dim * 2, fusion_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_dim, fusion_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.delta_head = nn.Conv2d(fusion_dim, 1, kernel_size=1, stride=1, padding=0)
        self.gate_head = nn.Conv2d(fusion_dim, 1, kernel_size=1, stride=1, padding=0)

        # Start from the DA3 metric prediction and learn small residual updates.
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -2.0)

    def forward(
        self,
        da3_pre_head_feat: torch.Tensor,
        adamvs_feat: torch.Tensor,
        base_metric_depth: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        target_hw = da3_pre_head_feat.shape[-2:]
        if adamvs_feat.shape[-2:] != target_hw:
            adamvs_feat = F.interpolate(
                adamvs_feat,
                size=target_hw,
                mode="bilinear",
                align_corners=True,
            )

        # Lazy-adapt projection convs if channel dims mismatch
        if self.da3_proj.in_channels != da3_pre_head_feat.shape[1]:
            self.da3_proj = nn.Conv2d(da3_pre_head_feat.shape[1], self.fusion_dim, kernel_size=1).to(da3_pre_head_feat.device)
        if self.ada_proj.in_channels != adamvs_feat.shape[1]:
            self.ada_proj = nn.Conv2d(adamvs_feat.shape[1], self.fusion_dim, kernel_size=1).to(adamvs_feat.device)

        da3_proj = self.da3_proj(da3_pre_head_feat)
        ada_proj = self.ada_proj(adamvs_feat)
        fused_feat = self.fuse(torch.cat([da3_proj, ada_proj], dim=1))

        delta = self.delta_head(fused_feat).squeeze(1)
        gate = torch.sigmoid(self.gate_head(fused_feat).squeeze(1))
        fused_metric = torch.clamp(base_metric_depth + gate * delta, min=1e-3)
        return fused_metric, {
            "fusion_delta": delta,
            "fusion_gate": gate,
        }


class CrossAttentionFusionHead(nn.Module):
    """Multi-scale cross-attention fusion for DA3 and Ada-MVS depth features.

    The head fuses two DA3 features and two Ada-MVS pre-depth features, which
    makes it closer to the requested "前一层或者几层特征" fusion recipe.
    """

    def __init__(self, da3_in_dims, ada_in_dims, fusion_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.fusion_dim = fusion_dim
        if isinstance(da3_in_dims, int):
            da3_in_dims = [da3_in_dims, da3_in_dims]
        if isinstance(ada_in_dims, int):
            ada_in_dims = [ada_in_dims, ada_in_dims]
        self.da3_proj = nn.ModuleList([nn.Conv2d(dim, fusion_dim, kernel_size=1) for dim in da3_in_dims])
        self.ada_proj = nn.ModuleList([nn.Conv2d(dim, fusion_dim, kernel_size=1) for dim in ada_in_dims])

        self.high_da3_attn = nn.MultiheadAttention(fusion_dim, num_heads=num_heads, batch_first=True)
        self.high_ada_attn = nn.MultiheadAttention(fusion_dim, num_heads=num_heads, batch_first=True)
        self.low_da3_attn = nn.MultiheadAttention(fusion_dim, num_heads=num_heads, batch_first=True)
        self.low_ada_attn = nn.MultiheadAttention(fusion_dim, num_heads=num_heads, batch_first=True)

        self.high_da3_norm = nn.LayerNorm(fusion_dim)
        self.high_ada_norm = nn.LayerNorm(fusion_dim)
        self.low_da3_norm = nn.LayerNorm(fusion_dim)
        self.low_ada_norm = nn.LayerNorm(fusion_dim)

        self.refine = nn.Sequential(
            nn.Conv2d(fusion_dim * 4, fusion_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_dim, fusion_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.delta_head = nn.Conv2d(fusion_dim, 1, kernel_size=1)
        self.gate_head = nn.Conv2d(fusion_dim, 1, kernel_size=1)
        self.residual_gate = nn.Conv2d(fusion_dim, 1, kernel_size=1)

        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -2.0)
        nn.init.zeros_(self.residual_gate.weight)
        nn.init.constant_(self.residual_gate.bias, -1.0)

    def _attn_hw(self, target_hw: Tuple[int, int], scale: int) -> Tuple[int, int]:
        h, w = target_hw
        return max(4, h // scale), max(4, w // scale)

    def _cross_context(self, da3_feat: torch.Tensor, ada_feat: torch.Tensor, attn: nn.MultiheadAttention, reverse_attn: nn.MultiheadAttention, da3_norm: nn.LayerNorm, ada_norm: nn.LayerNorm, scale: int) -> torch.Tensor:
        target_hw = da3_feat.shape[-2:]
        att_h, att_w = self._attn_hw(target_hw, scale)
        da3_tokens = F.adaptive_avg_pool2d(da3_feat, (att_h, att_w)).flatten(2).transpose(1, 2)
        ada_tokens = F.adaptive_avg_pool2d(ada_feat, (att_h, att_w)).flatten(2).transpose(1, 2)

        da3_ctx, _ = attn(da3_norm(da3_tokens), ada_norm(ada_tokens), ada_norm(ada_tokens), need_weights=False)
        ada_ctx, _ = reverse_attn(ada_norm(ada_tokens), da3_norm(da3_tokens), da3_norm(da3_tokens), need_weights=False)

        da3_ctx = da3_ctx.transpose(1, 2).reshape(da3_feat.shape[0], -1, att_h, att_w)
        ada_ctx = ada_ctx.transpose(1, 2).reshape(ada_feat.shape[0], -1, att_h, att_w)
        return 0.5 * (da3_ctx + ada_ctx)

    def forward(self, da3_feats, adamvs_feats, base_metric_depth: torch.Tensor):
        if not isinstance(da3_feats, (list, tuple)):
            da3_feats = [da3_feats]
        if not isinstance(adamvs_feats, (list, tuple)):
            adamvs_feats = [adamvs_feats]
        if len(da3_feats) == 1:
            da3_feats = [da3_feats[0], da3_feats[0]]
        if len(adamvs_feats) == 1:
            adamvs_feats = [adamvs_feats[0], adamvs_feats[0]]

        da3_low, da3_high = da3_feats[0], da3_feats[-1]
        ada_low, ada_high = adamvs_feats[0], adamvs_feats[-1]
        target_hw = da3_high.shape[-2:]

        # Lazily adapt projection convs if input channel counts differ from initialized ones
        if self.da3_proj[0].in_channels != da3_low.shape[1]:
            self.da3_proj[0] = nn.Conv2d(da3_low.shape[1], self.fusion_dim, kernel_size=1).to(da3_low.device)
        if self.da3_proj[-1].in_channels != da3_high.shape[1]:
            self.da3_proj[-1] = nn.Conv2d(da3_high.shape[1], self.fusion_dim, kernel_size=1).to(da3_high.device)
        if self.ada_proj[0].in_channels != ada_low.shape[1]:
            self.ada_proj[0] = nn.Conv2d(ada_low.shape[1], self.fusion_dim, kernel_size=1).to(ada_low.device)
        if self.ada_proj[-1].in_channels != ada_high.shape[1]:
            self.ada_proj[-1] = nn.Conv2d(ada_high.shape[1], self.fusion_dim, kernel_size=1).to(ada_high.device)

        da3_low_p = self.da3_proj[0](F.interpolate(da3_low, size=target_hw, mode="bilinear", align_corners=True) if da3_low.shape[-2:] != target_hw else da3_low)
        da3_high_p = self.da3_proj[-1](da3_high)
        ada_low_p = self.ada_proj[0](F.interpolate(ada_low, size=target_hw, mode="bilinear", align_corners=True) if ada_low.shape[-2:] != target_hw else ada_low)
        ada_high_p = self.ada_proj[-1](F.interpolate(ada_high, size=target_hw, mode="bilinear", align_corners=True) if ada_high.shape[-2:] != target_hw else ada_high)

        high_ctx = self._cross_context(
            da3_high_p, ada_high_p,
            self.high_da3_attn, self.high_ada_attn,
            self.high_da3_norm, self.high_ada_norm,
            scale=4,
        )
        low_ctx = self._cross_context(
            da3_low_p, ada_low_p,
            self.low_da3_attn, self.low_ada_attn,
            self.low_da3_norm, self.low_ada_norm,
            scale=8,
        )

        if low_ctx.shape[-2:] != target_hw:
            low_ctx = F.interpolate(low_ctx, size=target_hw, mode="bilinear", align_corners=True)
        if high_ctx.shape[-2:] != target_hw:
            high_ctx = F.interpolate(high_ctx, size=target_hw, mode="bilinear", align_corners=True)

        residual = 0.5 * (da3_low_p + ada_low_p) + 0.5 * (da3_high_p + ada_high_p)
        fused = self.refine(torch.cat([da3_high_p, ada_high_p, high_ctx + residual, low_ctx], dim=1))

        delta = self.delta_head(fused).squeeze(1)
        gate = torch.sigmoid(self.gate_head(fused).squeeze(1) + 0.5 * self.residual_gate(fused).squeeze(1))

        if base_metric_depth.shape[-2:] != target_hw:
            base_metric_depth = F.interpolate(
                base_metric_depth.unsqueeze(1), size=target_hw, mode="bilinear", align_corners=True
            ).squeeze(1)
        fused_metric = torch.clamp(base_metric_depth + gate * delta, min=1e-3)
        return fused_metric, {
            "fusion_delta": delta,
            "fusion_gate": gate,
        }


class LayerAttentionFusionHead(CrossAttentionFusionHead):
    """Backward-compatible alias for the cross-attention fusion head."""

    pass


class DA3MVSForMetricDepth(nn.Module):
    def __init__(
        self,
        da3_model,
        scale_shift: MetricScaleShift,
        adamvs_encoder: AdaMVSPreDepthFeatureExtractor,
        fusion_head: DA3MVSFusionHead,
    ):
        super().__init__()
        self.da3 = da3_model
        self.scale_shift = scale_shift
        self.adamvs_encoder = adamvs_encoder
        self.fusion_head = fusion_head
        self._last_da3_low_feat: Optional[torch.Tensor] = None
        self._last_da3_high_feat: Optional[torch.Tensor] = None

        head = self.da3.model.head
        if not hasattr(head, "scratch") or not hasattr(head.scratch, "output_conv2"):
            raise RuntimeError("DA3 head does not expose scratch.output_conv2, cannot extract pre-head feature")
        self._low_feat_hook = head.scratch.output_conv1.register_forward_hook(self._cache_da3_low_feature)
        self._high_feat_hook = head.scratch.output_conv2[0].register_forward_hook(self._cache_da3_high_feature)

    def _cache_da3_low_feature(self, module, inputs, outputs):
        if isinstance(outputs, torch.Tensor):
            self._last_da3_low_feat = outputs

    def _cache_da3_high_feature(self, module, inputs, outputs):
        if inputs and isinstance(inputs[0], torch.Tensor):
            self._last_da3_high_feat = inputs[0]

    def forward(self, images_input, extrinsics=None, intrinsics=None, depth_gt=None, valid_mask=None):
        self._last_da3_low_feat = None
        self._last_da3_high_feat = None
        if extrinsics is not None and extrinsics.ndim == 3:
            extrinsics = extrinsics.unsqueeze(1)
        if intrinsics is not None and intrinsics.ndim == 3:
            intrinsics = intrinsics.unsqueeze(1)
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            if extrinsics is not None and intrinsics is not None:
                output = self.da3.model(images_input, extrinsics=extrinsics, intrinsics=intrinsics)
            else:
                output = self.da3.model(images_input)

            pred_depth = output.depth
            if pred_depth.ndim == 4 and pred_depth.shape[1] == 1:
                pred_depth = pred_depth[:, 0]
            elif pred_depth.ndim == 4:
                pred_depth = pred_depth[:, 0]

            pred_conf = output.get("depth_conf", None)
            if pred_conf is not None and pred_conf.ndim == 4 and pred_conf.shape[1] == 1:
                pred_conf = pred_conf[:, 0]

            pred_sky = output.get("sky", None)
            if pred_sky is not None and pred_sky.ndim == 4 and pred_sky.shape[1] == 1:
                pred_sky = pred_sky[:, 0]

            metric_residual = output.get("metric_residual", None)
            metric_log_scale = output.get("metric_log_scale", None)
            metric_shift = output.get("metric_shift", None)
            metric_depth = output.get("metric_depth", None)

        if self._last_da3_low_feat is None or self._last_da3_high_feat is None:
            raise RuntimeError("Failed to capture DA3 multi-scale pre-head features via hooks")

        pred_depth = pred_depth.float()
        if metric_depth is not None:
            pred_metric = metric_depth.float()
        else:
            pred_metric = self.scale_shift(pred_depth)
            if metric_log_scale is not None:
                pred_metric = pred_metric * torch.exp(metric_log_scale.float().view(-1, 1, 1))
            if metric_shift is not None:
                pred_metric = pred_metric + metric_shift.float().view(-1, 1, 1)
            if metric_residual is not None:
                pred_metric = pred_metric + metric_residual.float()

        B, S = images_input.shape[:2]
        da3_low_feat = self._last_da3_low_feat.view(B, S, *self._last_da3_low_feat.shape[1:])[:, 0].float()
        da3_high_feat = self._last_da3_high_feat.view(B, S, *self._last_da3_high_feat.shape[1:])[:, 0].float()
        if extrinsics is None or intrinsics is None or depth_gt is None:
            raise ValueError("Ada-MVS pre-depth fusion requires extrinsics, intrinsics, and depth_gt")
        adamvs_images = images_input.squeeze(1) if images_input.ndim == 6 else images_input
        adamvs_feats = self.adamvs_encoder(adamvs_images.float(), extrinsics, intrinsics, depth_gt, valid_mask)
        pred_metric_fused, fusion_aux = self.fusion_head(
            [da3_low_feat, da3_high_feat],
            [adamvs_feats["stage2"].float(), adamvs_feats["stage3"].float()],
            pred_metric,
        )

        aux_info = {
            "metric_log_scale": metric_log_scale,
            "metric_shift": metric_shift,
            "fusion_delta": fusion_aux["fusion_delta"],
            "fusion_gate": fusion_aux["fusion_gate"],
        }
        return pred_depth, pred_metric_fused, pred_conf, pred_sky, aux_info


def get_raw_model(model):
    while isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module
    return model


def count_params_safe(model: nn.Module, trainable_only: bool = False) -> int:
    total = 0
    for p in model.parameters():
        if trainable_only and not p.requires_grad:
            continue
        if isinstance(p, torch.nn.parameter.UninitializedParameter):
            continue
        total += p.numel()
    return total


def log_section_scalars(writer, section: str, metrics: Dict[str, float], step: int) -> None:
    if writer is None:
        return
    for name, value in metrics.items():
        if isinstance(value, (int, float, np.floating)):
            writer.add_scalar(f"{section}/{name}", float(value), step)


def _squeeze_spatial_map(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.ndim == 4:
        return tensor[:, 0]
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor[0]
    return tensor


def _safe_tensor_stats(values: torch.Tensor, prefix: str) -> Dict[str, float]:
    if values.numel() == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_median": 0.0,
        }
    stats = {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_max": float(values.max().item()),
        f"{prefix}_median": float(values.median().item()),
    }
    if values.numel() > 1:
        stats[f"{prefix}_std"] = float(values.std(unbiased=False).item())
    else:
        stats[f"{prefix}_std"] = 0.0
    return stats

def collect_depth_distribution(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, float]:
    pred_depth = _squeeze_spatial_map(pred_depth).detach()
    gt_depth = _squeeze_spatial_map(gt_depth).detach()
    valid_mask = _squeeze_spatial_map(valid_mask).detach().bool()

    # Normalize leading singleton batch dims: if one tensor is 2D and another is 3D with size[0]==1, squeeze the 3D to 2D
    if gt_depth.ndim == 2:
        if pred_depth.ndim == 3 and pred_depth.shape[0] == 1:
            pred_depth = pred_depth[0]
        if valid_mask.ndim == 3 and valid_mask.shape[0] == 1:
            valid_mask = valid_mask[0]
    if pred_depth.ndim == 2:
        if gt_depth.ndim == 3 and gt_depth.shape[0] == 1:
            gt_depth = gt_depth[0]
        if valid_mask.ndim == 3 and valid_mask.shape[0] == 1:
            valid_mask = valid_mask[0]

    valid_mask = valid_mask & torch.isfinite(pred_depth) & torch.isfinite(gt_depth) & (gt_depth > 1.0)
    if valid_mask.sum() < 10:
        return {
            "pred_depth_mean": 0.0,
            "pred_depth_std": 0.0,
            "pred_depth_min": 0.0,
            "pred_depth_max": 0.0,
            "pred_depth_median": 0.0,
            "gt_depth_mean": 0.0,
            "gt_depth_std": 0.0,
            "gt_depth_min": 0.0,
            "gt_depth_max": 0.0,
            "gt_depth_median": 0.0,
            "median_align_scale": 0.0,
        }

    pred_depth_valid = pred_depth[valid_mask]
    gt_valid = gt_depth[valid_mask]
    median_align_scale = float(gt_valid.median().item() / max(pred_depth_valid.median().item(), 1e-6))

    stats = {}
    stats.update(_safe_tensor_stats(pred_depth_valid, "pred_depth"))
    stats.update(_safe_tensor_stats(gt_valid, "gt_depth"))
    stats["median_align_scale"] = median_align_scale
    return stats


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr_value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr_value


def capture_rng_state() -> Dict[str, object]:
    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random_state"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, object]) -> None:
    if not state:
        return
    if "python_random_state" in state:
        random.setstate(state["python_random_state"])
    if "numpy_random_state" in state:
        np.random.set_state(state["numpy_random_state"])
    if "torch_random_state" in state:
        torch.set_rng_state(state["torch_random_state"])
    if torch.cuda.is_available() and "torch_cuda_random_state" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_random_state"])


def apply_lora_to_model(model, lora_rank=16, lora_alpha=32, lora_dropout=0.05, target_modules=None):
    from peft import LoraConfig, get_peft_model

    if target_modules is None:
        target_modules = ["qkv", "proj"]
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_da3_lora_checkpoint(model, lora_checkpoint: str):
    if not lora_checkpoint:
        return model

    ckpt = torch.load(lora_checkpoint, map_location="cpu", weights_only=False)
    model_sd = ckpt.get("model_state_dict", ckpt)

    logger.info("Loading DA3 LoRA warm start from %s", lora_checkpoint)
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if missing:
        logger.warning("Missing keys when loading DA3 LoRA warm start (%d): %s", len(missing), missing[:5])
    if unexpected:
        logger.warning("Unexpected keys when loading DA3 LoRA warm start (%d): %s", len(unexpected), unexpected[:5])
    return model


def build_da3_model(model_name="da3-large", pretrained_path=None):
    from da3.api import DepthAnything3

    model = DepthAnything3(model_name=model_name)
    if pretrained_path and os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded pretrained weights from %s", pretrained_path)
    return model


def init_scale_from_dataset(da3_model, dataset, device, max_samples=200):
    logger.info("Initializing scale from %d dataset samples...", min(max_samples, len(dataset)))
    da3_model.eval()
    pred_medians = []
    gt_medians = []
    sample_indices = np.linspace(0, len(dataset) - 1, min(max_samples, len(dataset)), dtype=int)
    for idx in sample_indices:
        try:
            sample = dataset[idx]
            images = sample["image"].unsqueeze(0).to(device)
            depth_gt = sample["depth_gt"].numpy()
            valid_mask = sample["valid_mask"].numpy().astype(bool)
            with torch.no_grad():
                images_input = images.unsqueeze(1)
                autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    output = da3_model.model(images_input)
                    pred = output.depth
                pred = pred.float().cpu().numpy()
            pred = np.squeeze(pred)
            if pred.shape != depth_gt.shape:
                from PIL import Image as PILImage
                h, w = depth_gt.shape
                pred = np.array(PILImage.fromarray(pred).resize((w, h), PILImage.BILINEAR))
            valid = valid_mask & np.isfinite(pred) & np.isfinite(depth_gt) & (depth_gt > 1.0) & (pred > 1e-6)
            if valid.sum() < 100:
                continue
            pred_med = float(np.median(pred[valid]))
            gt_med = float(np.median(depth_gt[valid]))
            pred_medians.append(pred_med)
            gt_medians.append(gt_med)
        except Exception as e:
            logger.warning("Skipping sample %d: %s", idx, e)
            continue

    if len(pred_medians) < 10:
        logger.warning("Not enough valid samples for scale init, using 1.0")
        return 1.0

    pred_medians = np.array(pred_medians)
    gt_medians = np.array(gt_medians)
    init_scale = float(np.median(gt_medians / np.maximum(pred_medians, 1e-6)))
    logger.info("Init scale: %.4f (from %d samples)", init_scale, len(pred_medians))
    return init_scale
@torch.no_grad()
def validate(model, dataloader, criterion, device, global_step, writer, tag="val"):
    raw = get_raw_model(model)
    raw.eval()
    total_loss = 0.0
    total_abs_rel = 0.0
    total_rmse = 0.0
    total_mae = 0.0
    total_scale = 0.0
    num_batches = 0
    num_valid_samples = 0
    logged_depth_images = False

    for batch in dataloader:
        images = batch["image"].to(device)
        depth_gt = batch["depth_gt"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        extrinsics = batch["extrinsics"].to(device)
        intrinsics = batch["intrinsics"].to(device)
        images_input = images.unsqueeze(1)

        pred_depth, pred_metric, pred_conf, pred_sky, aux_out = raw(
            images_input,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            depth_gt=depth_gt,
            valid_mask=valid_mask,
        )

        if pred_depth.shape[-2:] != depth_gt.shape[-2:]:
            pred_depth = F.interpolate(
                pred_depth.unsqueeze(1), size=depth_gt.shape[-2:],
                mode="bilinear", align_corners=True,
            ).squeeze(1)
        if pred_metric.shape[-2:] != depth_gt.shape[-2:]:
            pred_metric = F.interpolate(
                pred_metric.unsqueeze(1), size=depth_gt.shape[-2:],
                mode="bilinear", align_corners=True,
            ).squeeze(1)
        if pred_conf is not None and pred_conf.shape[-2:] != depth_gt.shape[-2:]:
            pred_conf = F.interpolate(
                pred_conf.unsqueeze(1), size=depth_gt.shape[-2:],
                mode="bilinear", align_corners=True,
            ).squeeze(1)
        if pred_sky is not None and pred_sky.shape[-2:] != depth_gt.shape[-2:]:
            pred_sky = F.interpolate(
                pred_sky.unsqueeze(1), size=depth_gt.shape[-2:],
                mode="bilinear", align_corners=True,
            ).squeeze(1)

        loss_dict = criterion(
            pred_depth,
            pred_metric,
            depth_gt,
            valid_mask,
            pred_conf,
            pred_sky,
            log_scale=aux_out.get("metric_log_scale") if isinstance(aux_out, dict) else None,
            shift=aux_out.get("metric_shift") if isinstance(aux_out, dict) else None,
        )
        total_loss += loss_dict["loss"].item()

        if not logged_depth_images and writer is not None:
            logged_depth_images = True
            n_vis = min(images.shape[0], 2)
            for vi in range(n_vis):
                gt_vis = _squeeze_spatial_map(depth_gt[vi]).cpu().numpy()
                pred_vis = _squeeze_spatial_map(pred_metric[vi]).cpu().numpy()
                mask_vis = _squeeze_spatial_map(valid_mask[vi]).cpu().numpy().astype(bool)
                gt_valid = gt_vis[mask_vis]
                pred_valid = pred_vis[mask_vis]
                scale = np.median(gt_valid) / max(np.median(pred_valid), 1e-6) if pred_valid.size > 0 else 1.0
                pred_vis = pred_vis * scale

                def _norm(d, m):
                    if d.ndim == 3 and d.shape[0] == 1:
                        d = d[0]
                    if m.ndim == 3 and m.shape[0] == 1:
                        m = m[0]
                    v = d[m]
                    mx = np.abs(v).max() if v.size > 0 else 1.0
                    return d / max(mx, 1e-6)

                writer.add_image(
                    f"{tag}/depth/gt_{vi}",
                    torch.from_numpy(_norm(gt_vis, mask_vis)).unsqueeze(0),
                    global_step,
                )
                writer.add_image(
                    f"{tag}/depth/pred_{vi}",
                    torch.from_numpy(_norm(pred_vis, mask_vis)).unsqueeze(0),
                    global_step,
                )
                err_map = np.abs(pred_vis - gt_vis)
                gt_max = np.abs(gt_valid).max() if gt_valid.size > 0 else 1.0
                writer.add_image(
                    f"{tag}/depth/error_{vi}",
                    torch.from_numpy(err_map / max(gt_max, 1e-6)).unsqueeze(0),
                    global_step,
                )

        pred_np = pred_metric.cpu().numpy()
        gt_np = depth_gt.cpu().numpy()
        mask_np = valid_mask.cpu().numpy().astype(bool)
        if pred_np.ndim == 4 and pred_np.shape[1] == 1:
            pred_np = pred_np[:, 0]
        if gt_np.ndim == 4 and gt_np.shape[1] == 1:
            gt_np = gt_np[:, 0]
        if mask_np.ndim == 4 and mask_np.shape[1] == 1:
            mask_np = mask_np[:, 0]
        for i in range(pred_np.shape[0]):
            m = mask_np[i] & np.isfinite(pred_np[i]) & np.isfinite(gt_np[i]) & (gt_np[i] > 1.0)
            if m.sum() < 10:
                continue
            p = pred_np[i][m]
            g = gt_np[i][m]
            scale = np.median(g) / max(np.median(p), 1e-6)
            p = p * scale
            total_scale += float(scale)
            total_abs_rel += np.mean(np.abs(p - g) / np.maximum(g, 1e-6))
            total_rmse += np.sqrt(np.mean((p - g) ** 2))
            total_mae += np.mean(np.abs(p - g))
            num_valid_samples += 1
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    n_valid = max(num_valid_samples, 1)
    metrics = {
        "loss": avg_loss,
        "abs_rel": total_abs_rel / n_valid,
        "rmse": total_rmse / n_valid,
        "mae": total_mae / n_valid,
        "median_scale": total_scale / n_valid,
    }
    if writer is not None:
        log_section_scalars(writer, tag, metrics, global_step)
    return metrics


def set_trainable_parts(model):
    raw = get_raw_model(model)
    da3_model = raw.da3.model

    for param in da3_model.parameters():
        param.requires_grad = False
    if hasattr(raw, "scale_shift"):
        for param in raw.scale_shift.parameters():
            param.requires_grad = True

    head_modules = [getattr(da3_model, "head", None), getattr(da3_model, "cam_enc", None), getattr(da3_model, "cam_dec", None)]
    for module in head_modules:
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True

    for name, param in da3_model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    for p in raw.fusion_head.parameters():
        p.requires_grad = True

    for p in raw.adamvs_encoder.parameters():
        p.requires_grad = False


def main():
    parser = argparse.ArgumentParser(
        description="Train DA3MVS with Ada-MVS feature fusion and metric supervision on WHU-OMVS"
    )
    parser.add_argument("--dataset_root", type=str, default="dataset/WHU-OMVS")
    parser.add_argument("--output_dir", type=str, default="exp/whu-omvs/train_da3mvs/da3_large_adamvs_fusion")
    parser.add_argument("--model_name", type=str, default="da3-large",
                        choices=["da3-base", "da3-large", "da3-giant", "da3-small"])
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--da3_lora_checkpoint", type=str, default=None)
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=4e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=800)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="+", default=["qkv", "proj"])
    parser.add_argument("--adapter_hidden_dim", type=int, default=64)
    parser.add_argument("--adapter_depth_norm", type=float, default=600.0)
    parser.add_argument(
        "--adamvs_ckpt",
        type=str,
        default="weights/adamvs/adamvs_whuomvs/model_000019_0.1339.ckpt",
    )
    parser.add_argument("--fusion_dim", type=int, default=128)
    parser.add_argument("--fusion_type", type=str, default="cross_attention", choices=["conv", "cross_attention", "layer_attention"])
    parser.add_argument("--loss_profile", type=str, default="metric_v4", choices=["metric_v4"])
    parser.add_argument("--relative_si_weight", type=float, default=0.3)
    parser.add_argument("--metric_si_weight", type=float, default=1.2)
    parser.add_argument("--logl1_weight", type=float, default=4.0)
    parser.add_argument("--l1_weight", type=float, default=0.7)
    parser.add_argument("--absrel_weight", type=float, default=0.2)
    parser.add_argument("--gradient_weight", type=float, default=0.15)
    parser.add_argument("--range_weight", type=float, default=0.02)
    parser.add_argument("--confidence_weight", type=float, default=0.1)
    parser.add_argument("--sky_weight", type=float, default=0.05)
    parser.add_argument("--confidence_tau", type=float, default=120.0)
    parser.add_argument("--sky_threshold", type=float, default=0.3)
    parser.add_argument("--far_depth_start", type=float, default=350.0)
    parser.add_argument("--far_depth_boost", type=float, default=0.35)
    parser.add_argument("--far_depth_transition", type=float, default=80.0)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_val_samples", type=int, default=-1)
    parser.add_argument("--max_test_samples", type=int, default=-1)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--save_every_n_epochs", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=0.8)
    parser.add_argument("--ema_alpha", type=float, default=0.2)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.003)
    parser.add_argument("--early_stop_absrel_min_delta", type=float, default=0.0005)
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Using GPU: %s (CUDA_VISIBLE_DEVICES=%s)", gpu_ids, os.environ["CUDA_VISIBLE_DEVICES"])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "logs"))

    logger.info("Building DA3 model: %s", args.model_name)
    da3_model = build_da3_model(args.model_name, args.pretrained_path)
    da3_model = da3_model.to(device)

    dataset_root = str(
        Path(args.dataset_root).resolve() if not os.path.isabs(args.dataset_root) else args.dataset_root
    )
    train_areas = ["area1", "area4", "area5", "area6"]
    test_areas = ["area2", "area3"]

    train_dataset = WHUOMVSDepthDataset(
        dataset_root=dataset_root, split="train", areas=train_areas,
        process_res=args.process_res, augment=True,
        max_samples=args.max_train_samples if args.max_train_samples > 0 else -1,
    )
    val_dataset = WHUOMVSDepthDataset(
        dataset_root=dataset_root, split="train", areas=train_areas[:1],
        process_res=args.process_res, augment=False,
        max_samples=args.max_val_samples if args.max_val_samples > 0 else 500,
    )
    test_dataset = WHUOMVSDepthDataset(
        dataset_root=dataset_root, split="test", areas=test_areas,
        process_res=args.process_res, augment=False,
        max_samples=args.max_test_samples if args.max_test_samples > 0 else 500,
    )

    scale_shift = MetricScaleShift(init_scale=1.0, init_shift=0.0).to(device)
    adamvs_ckpt = str(Path(args.adamvs_ckpt).resolve()) if args.adamvs_ckpt else ""
    adamvs_encoder = AdaMVSPreDepthFeatureExtractor(
        ckpt_path=adamvs_ckpt,
        trainable=False,
    ).to(device)
    if args.fusion_type in {"cross_attention", "layer_attention"}:
        fusion_head = CrossAttentionFusionHead(
            da3_in_dims=[64, 64],
            ada_in_dims=[16, 8],
            fusion_dim=args.fusion_dim,
        ).to(device)
    else:
        fusion_head = DA3MVSFusionHead(
            da3_in_dim=64,
            ada_in_dim=8,
            fusion_dim=args.fusion_dim,
        ).to(device)

    model = DA3MVSForMetricDepth(da3_model, scale_shift, adamvs_encoder, fusion_head)

    logger.info("Applying LoRA: rank=%d, alpha=%d, targets=%s",
                args.lora_rank, args.lora_alpha, args.lora_target_modules)
    da3_inner = da3_model.model
    da3_inner = apply_lora_to_model(
        da3_inner, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=args.lora_target_modules,
    )
    da3_model.model = da3_inner

    if args.da3_lora_checkpoint:
        model = load_da3_lora_checkpoint(model, args.da3_lora_checkpoint)

    set_trainable_parts(model)

    raw_model = get_raw_model(model)
    head_params = count_params_safe(raw_model.da3.model.head)
    trainable_params = count_params_safe(model, trainable_only=True)
    total_params = count_params_safe(model)
    logger.info("Trainable: %d / %d (%.2f%%) | Head: %d | Frozen helper: %d",
                trainable_params, total_params, 100.0 * trainable_params / total_params,
                head_params, sum(p.numel() for p in raw_model.scale_shift.parameters()))
    logger.info("Metric fusion mode with Ada-MVS pre-depth feature fusion enabled.")

    criterion = MetricDepthLossV4(
        relative_si_weight=args.relative_si_weight,
        metric_si_weight=args.metric_si_weight,
        logl1_weight=args.logl1_weight,
        l1_weight=args.l1_weight,
        absrel_weight=args.absrel_weight,
        gradient_weight=args.gradient_weight,
        range_weight=args.range_weight,
        confidence_weight=args.confidence_weight,
        sky_weight=args.sky_weight,
        scale_reg_weight=0.005,
        shift_reg_weight=0.005,
        depth_range=(1.0, args.adapter_depth_norm * 3.5),
        depth_norm=args.adapter_depth_norm,
        confidence_tau=args.confidence_tau,
        sky_threshold=args.sky_threshold,
        far_depth_start=args.far_depth_start,
        far_depth_boost=args.far_depth_boost,
        far_depth_transition=args.far_depth_transition,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(train_loader) // args.grad_accum_steps
    warmup_steps = min(args.warmup_steps, max(total_steps - 1, 1))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    best_val_mae = float("inf")
    best_test_mae = float("inf")
    best_ema_val_mae = float("inf")
    ema_val_mae: Optional[float] = None
    early_stop_wait = 0
    global_step = 0

    if args.resume and os.path.exists(args.resume):
        logger.info("Resuming from checkpoint: %s", args.resume)
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "global_step" in ckpt:
            global_step = ckpt["global_step"]
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"])
        if "best_val_mae" in ckpt:
            best_val_mae = ckpt["best_val_mae"]
        if "best_test_mae" in ckpt:
            best_test_mae = ckpt["best_test_mae"]
        if "best_ema_val_mae" in ckpt:
            best_ema_val_mae = ckpt["best_ema_val_mae"]
        if "ema_val_mae" in ckpt:
            ema_val_mae = ckpt["ema_val_mae"]
        if "early_stop_wait" in ckpt:
            early_stop_wait = int(ckpt["early_stop_wait"])
        if "rng_state" in ckpt:
            restore_rng_state(ckpt["rng_state"])
        logger.info(
            "Resumed from epoch=%d global_step=%d best_val_mae=%.4f best_test_mae=%.4f",
            start_epoch, global_step, best_val_mae, best_test_mae,
        )

    logger.info("=" * 70)
    logger.info("Training config:")
    logger.info("  Model: %s (%.1fM params)", args.model_name, total_params / 1e6)
    logger.info("  Train: %d samples, Val: %d samples, Test: %d samples",
                len(train_dataset), len(val_dataset), len(test_dataset))
    logger.info("  Epochs: %d (single-stage metric fusion)", args.epochs)
    logger.info("  LR: %.2e, Warmup: %d steps, Grad accum: %d",
                args.lr, warmup_steps, args.grad_accum_steps)
    logger.info("  Loss profile: %s | DA3 pre-head + Ada-MVS pre-depth fusion", args.loss_profile)
    logger.info("  LoRA: rank=%d, alpha=%d", args.lora_rank, args.lora_alpha)
    logger.info("  Ada-MVS ckpt: %s | fusion_dim=%d", adamvs_ckpt, args.fusion_dim)
    logger.info("  Loss: si=%.1f logl1=%.1f l1=%.1f grad=%.2f range=%.2f",
                args.relative_si_weight, args.logl1_weight, args.l1_weight,
                args.gradient_weight, args.range_weight)
    logger.info("  Grad clip: max_norm=%.2f | EMA alpha=%.2f | EarlyStop patience=%d mae_delta=%.4f",
                args.max_grad_norm, args.ema_alpha, args.early_stop_patience,
                args.early_stop_min_delta)
    logger.info("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        sched_lr = scheduler.get_last_lr()[0]
        set_optimizer_lr(optimizer, sched_lr)

        trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info("=" * 70)
        logger.info("Epoch %d/%d | trainable=%d lr=%.2e (sched=%.2e)",
                    epoch + 1, args.epochs, trainable_now, current_lr, sched_lr)
        logger.info("=" * 70)

        model.train()

        optimizer.zero_grad()
        epoch_loss = 0.0
        epoch_l1 = 0.0
        num_batches = 0

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            depth_gt = batch["depth_gt"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            extrinsics = batch["extrinsics"].to(device)
            intrinsics = batch["intrinsics"].to(device)
            images_input = images.unsqueeze(1)

            pred_depth, pred_metric, _, _, aux_out = model(
                images_input,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                depth_gt=depth_gt,
                valid_mask=valid_mask,
            )

            if pred_depth.shape[-2:] != depth_gt.shape[-2:]:
                pred_depth = F.interpolate(
                    pred_depth.unsqueeze(1), size=depth_gt.shape[-2:],
                    mode="bilinear", align_corners=True,
                ).squeeze(1)
            if pred_metric.shape[-2:] != depth_gt.shape[-2:]:
                pred_metric = F.interpolate(
                    pred_metric.unsqueeze(1), size=depth_gt.shape[-2:],
                    mode="bilinear", align_corners=True,
                ).squeeze(1)

            loss_dict = criterion(
                pred_depth,
                pred_metric,
                depth_gt,
                valid_mask,
                None,
                None,
                log_scale=aux_out.get("metric_log_scale") if isinstance(aux_out, dict) else None,
                shift=aux_out.get("metric_shift") if isinstance(aux_out, dict) else None,
            )
            loss = loss_dict["loss"] / args.grad_accum_steps
            loss.backward()
            depth_stats = collect_depth_distribution(pred_metric, depth_gt, valid_mask)

            if (step + 1) % args.grad_accum_steps == 0:
                if args.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], max_norm=args.max_grad_norm
                    )
                else:
                    grad_norm = torch.tensor(0.0, device=device)
                optimizer.step()
                scheduler.step()
                next_sched_lr = scheduler.get_last_lr()[0]
                set_optimizer_lr(optimizer, next_sched_lr)
                optimizer.zero_grad()
                global_step += 1

                current_lr = optimizer.param_groups[0]["lr"]
                train_log = {
                    "loss_total": loss_dict["loss"].item(),
                    "relative_si_loss": loss_dict["relative_si_loss"].item(),
                    "metric_si_loss": loss_dict["metric_si_loss"].item(),
                    "logl1_loss": loss_dict["logl1_loss"].item(),
                    "l1_loss": loss_dict["l1_loss"].item(),
                    "absrel_loss": loss_dict["absrel_loss"].item(),
                    "gradient_loss": loss_dict["gradient_loss"].item(),
                    "range_loss": loss_dict["range_loss"].item(),
                    "confidence_loss": loss_dict["confidence_loss"].item(),
                    "sky_loss": loss_dict["sky_loss"].item(),
                    "grad_norm": float(grad_norm.item()),
                }
                if aux_out.get("fusion_gate", None) is not None:
                    train_log["fusion_gate_mean"] = float(aux_out["fusion_gate"].mean().item())
                if aux_out.get("fusion_delta", None) is not None:
                    train_log["fusion_delta_mean"] = float(aux_out["fusion_delta"].mean().item())
                train_log.update(depth_stats)
                log_section_scalars(writer, "train", train_log, global_step)
                log_section_scalars(writer, "lr", {
                    "main": current_lr,
                    "scheduler": next_sched_lr,
                }, global_step)
                log_section_scalars(writer, "epoch", {"current": epoch + 1}, global_step)

            epoch_loss += loss_dict["loss"].item()
            epoch_l1 += loss_dict["l1_loss"].item()
            num_batches += 1

            if step % 500 == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    "  E%d/%d Step %d/%d | lr=%.2e loss=%.2f rel_si=%.4f metric_si=%.4f logl1=%.4f l1=%.2f grad=%.4f median_scale=%.4f gate=%.3f",
                    epoch + 1, args.epochs, step, len(train_loader),
                    current_lr, loss_dict["loss"].item(), loss_dict["relative_si_loss"].item(),
                    loss_dict["metric_si_loss"].item(),
                    loss_dict["logl1_loss"].item(), loss_dict["l1_loss"].item(),
                    loss_dict["gradient_loss"].item(), depth_stats.get("median_align_scale", 0.0),
                    float(aux_out["fusion_gate"].mean().item()) if aux_out.get("fusion_gate", None) is not None else 0.0,
                )

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        avg_epoch_l1 = epoch_l1 / max(num_batches, 1)
        logger.info("Epoch %d done | avg_loss=%.4f avg_l1=%.2f", epoch + 1, avg_epoch_loss, avg_epoch_l1)
        log_section_scalars(writer, "epoch", {
            "loss": avg_epoch_loss,
            "l1": avg_epoch_l1,
            "current": epoch + 1,
        }, epoch)

        val_metrics = validate(model, val_loader, criterion, device, global_step, writer, tag="val")
        test_metrics = validate(model, test_loader, criterion, device, global_step, writer, tag="test")
        logger.info(
            "  [EpochVal] E%d | val_mae=%.2f val_rmse=%.2f test_mae=%.2f test_rmse=%.2f val_scale=%.4f",
            epoch + 1,
            val_metrics["mae"], val_metrics["rmse"],
            test_metrics["mae"], test_metrics["rmse"],
            val_metrics.get("median_scale", 0.0),
        )

        if ema_val_mae is None:
            ema_val_mae = float(val_metrics["mae"])
        else:
            ema_val_mae = args.ema_alpha * float(val_metrics["mae"]) + (1.0 - args.ema_alpha) * ema_val_mae

        log_section_scalars(writer, "val", {
            "mae_ema": ema_val_mae,
        }, epoch + 1)

        mae_improved = ema_val_mae < best_ema_val_mae - args.early_stop_min_delta

        if mae_improved:
            best_ema_val_mae = ema_val_mae

        if mae_improved:
            early_stop_wait = 0
        else:
            early_stop_wait += 1
        logger.info(
            "  [EMA] E%d | mae_ema=%.4f(best=%.4f) wait=%d/%d",
            epoch + 1, ema_val_mae, best_ema_val_mae,
            early_stop_wait, args.early_stop_patience,
        )

        is_best = val_metrics["mae"] < best_val_mae
        if is_best:
            best_val_mae = val_metrics["mae"]
            best_test_mae = test_metrics["mae"]

        ckpt_path = ckpt_dir / f"epoch_{epoch+1:03d}.pt"
        checkpoint = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_mae": val_metrics["mae"],
            "test_mae": test_metrics["mae"],
            "best_val_mae": best_val_mae,
            "best_test_mae": best_test_mae,
            "best_ema_val_mae": best_ema_val_mae,
            "ema_val_mae": ema_val_mae,
            "early_stop_wait": early_stop_wait,
            "rng_state": capture_rng_state(),
            "args": vars(args),
        }
        torch.save(checkpoint, str(ckpt_path))
        latest_path = ckpt_dir / "latest.pt"
        torch.save(checkpoint, str(latest_path))
        logger.info("Saved checkpoint: %s (epoch=%d val_mae=%.4f test_mae=%.4f)",
                    ckpt_path, epoch + 1, val_metrics["mae"], test_metrics["mae"])

        if is_best:
            best_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_mae": val_metrics["mae"],
                "test_mae": test_metrics["mae"],
                "best_val_mae": best_val_mae,
                "best_test_mae": best_test_mae,
                "best_ema_val_mae": best_ema_val_mae,
                "ema_val_mae": ema_val_mae,
                "early_stop_wait": early_stop_wait,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "rng_state": capture_rng_state(),
                "args": vars(args),
            }, str(best_path))
            logger.info("New best model! val_mae=%.4f test_mae=%.4f", val_metrics["mae"], test_metrics["mae"])

        if args.early_stop_patience > 0 and early_stop_wait >= args.early_stop_patience:
            logger.info(
                "Early stopping triggered at epoch %d (wait=%d, best_ema=%.4f)",
                epoch + 1, early_stop_wait, best_ema_val_mae,
            )
            break

    lora_path = ckpt_dir / "lora_final.pt"
    lora_state = {
        k: v for k, v in model.state_dict().items()
        if "lora" in k.lower()
        or ".head." in k.lower()
        or "cam_enc" in k.lower()
        or "cam_dec" in k.lower()
        or "fusion_head" in k.lower()
        or "scale_shift" in k.lower()
    }
    torch.save(lora_state, str(lora_path))
    logger.info("Saved LoRA + fusion head weights: %s (%d keys)", lora_path, len(lora_state))

    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "best_val_mae": best_val_mae,
            "best_test_mae": best_test_mae,
            "best_ema_val_mae": best_ema_val_mae,
            "final_ema_val_mae": ema_val_mae,
            "args": vars(args),
        }, f, indent=2)

    writer.close()
    logger.info("Training complete. Best val_mae=%.4f, best_test_mae=%.4f", best_val_mae, best_test_mae)


if __name__ == "__main__":
    main()
