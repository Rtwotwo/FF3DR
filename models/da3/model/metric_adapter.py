from __future__ import annotations

import torch
import torch.nn as nn


class MetricAdapterV3(nn.Module):
    """Lightweight metric adapter for absolute depth calibration.

    Produces:
      - per-pixel residual correction in metric units
      - per-image log-scale and shift offsets
    """

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 64,
        depth_norm: float = 600.0,
    ) -> None:
        super().__init__()
        self.depth_norm = float(depth_norm)

        self.residual_conv1 = nn.Conv2d(feat_dim + 1, hidden_dim, 3, 1, 1)
        self.residual_conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
        self.residual_conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
        self.residual_out = nn.Conv2d(hidden_dim, 1, 1, 1, 0)

        self.scale_head = nn.Sequential(
            nn.Linear(feat_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

        self.act = nn.GELU()
        nn.init.zeros_(self.residual_out.weight)
        nn.init.zeros_(self.residual_out.bias)
        nn.init.zeros_(self.scale_head[-1].weight)
        nn.init.zeros_(self.scale_head[-1].bias)

    def forward(
        self,
        fused_feat: torch.Tensor,
        relative_depth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Residual map (scaled to metric depth units)
        depth_normed = relative_depth / max(self.depth_norm, 1.0)
        x = torch.cat([fused_feat, depth_normed.unsqueeze(1)], dim=1)
        x = self.act(self.residual_conv1(x))
        x = self.act(self.residual_conv2(x))
        x = self.act(self.residual_conv3(x))
        residual = self.residual_out(x).squeeze(1) * self.depth_norm

        # Per-image scale/shift offsets from pooled features + depth stats
        pooled = fused_feat.mean(dim=(2, 3))
        depth_mean = relative_depth.mean(dim=(1, 2))
        depth_std = relative_depth.std(dim=(1, 2))
        stats = torch.stack([depth_mean, depth_std], dim=1)
        scale_in = torch.cat([pooled, stats], dim=1)
        delta_log_scale, delta_shift = self.scale_head(scale_in).chunk(2, dim=1)
        delta_shift = delta_shift.squeeze(1) * self.depth_norm
        delta_log_scale = delta_log_scale.squeeze(1)

        return residual, delta_log_scale, delta_shift
