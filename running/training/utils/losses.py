from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DepthLoss(nn.Module):
    """
    Combined depth loss for supervised depth estimation.

    Components:
    - L1 loss on depth values
    - Scale-invariant log loss (Eigen et al.)
    - Edge-aware smoothness loss
    """

    def __init__(
        self,
        weight_l1=1.0,
        weight_silog=0.5,
        weight_smooth=0.1,
        max_depth=1.0,
    ):
        super().__init__()
        self.weight_l1 = weight_l1
        self.weight_silog = weight_silog
        self.weight_smooth = weight_smooth
        self.max_depth = max_depth

    def forward(self, pred_depth, gt_depth, image=None, mask=None):
        if gt_depth.dim() == 4 and gt_depth.shape[1] == 1:
            gt_depth_2d = gt_depth.squeeze(1)
        else:
            gt_depth_2d = gt_depth

        if pred_depth.dim() == 4 and pred_depth.shape[1] == 1:
            pred_depth_2d = pred_depth.squeeze(1)
        else:
            pred_depth_2d = pred_depth

        if mask is None:
            valid = (gt_depth_2d > 1e-3) & (gt_depth_2d < self.max_depth)
        else:
            if mask.dim() == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)
            valid = (mask > 0.5) & (gt_depth_2d > 1e-3) & (gt_depth_2d < self.max_depth)

        losses = {}

        if valid.sum() == 0:
            losses["total"] = torch.tensor(0.0, device=pred_depth.device, requires_grad=True)
            return losses

        pred_valid = pred_depth_2d[valid]
        gt_valid = gt_depth_2d[valid]

        if self.weight_l1 > 0:
            losses["l1"] = self.weight_l1 * F.l1_loss(pred_valid, gt_valid)

        if self.weight_silog > 0:
            log_diff = torch.log(pred_valid.clamp(min=1e-6)) - torch.log(gt_valid.clamp(min=1e-6))
            losses["silog"] = self.weight_silog * (
                log_diff.pow(2).mean() - 0.5 * log_diff.mean().pow(2)
            )

        if self.weight_smooth > 0 and image is not None:
            losses["smooth"] = self.weight_smooth * self._edge_aware_smoothness(
                pred_depth_2d.unsqueeze(1), image
            )

        losses["total"] = sum(losses.values())
        return losses

    def _edge_aware_smoothness(self, depth, image):
        mean_depth = depth.mean(dim=1, keepdim=True)
        mean_image = image.mean(dim=1, keepdim=True)

        depth_grad_x = torch.abs(mean_depth[:, :, :, :-1] - mean_depth[:, :, :, 1:])
        depth_grad_y = torch.abs(mean_depth[:, :, :-1, :] - mean_depth[:, :, 1:, :])

        image_grad_x = torch.abs(mean_image[:, :, :, :-1] - mean_image[:, :, :, 1:])
        image_grad_y = torch.abs(mean_image[:, :, :-1, :] - mean_image[:, :, 1:, :])

        weight_x = torch.exp(-image_grad_x)
        weight_y = torch.exp(-image_grad_y)

        smooth_x = (depth_grad_x * weight_x).mean()
        smooth_y = (depth_grad_y * weight_y).mean()

        return smooth_x + smooth_y


class PoseLoss(nn.Module):
    """
    Camera pose estimation loss.

    Components:
    - Rotation loss (geodesic distance on SO3)
    - Translation loss (L1 after scale alignment)
    """

    def __init__(
        self,
        weight_rotation=1.0,
        weight_translation=1.0,
    ):
        super().__init__()
        self.weight_rotation = weight_rotation
        self.weight_translation = weight_translation

    def forward(self, pred_extrinsics, gt_extrinsics):
        losses = {}

        pred_R = pred_extrinsics[:, :, :3, :3]
        pred_t = pred_extrinsics[:, :, :3, 3]

        gt_R = gt_extrinsics[:, :, :3, :3]
        gt_t = gt_extrinsics[:, :, :3, 3]

        if self.weight_rotation > 0:
            losses["rotation"] = self.weight_rotation * self._geodesic_loss(pred_R, gt_R)

        if self.weight_translation > 0:
            pred_t_norm = pred_t / (pred_t.norm(dim=-1, keepdim=True).clamp(min=1e-6))
            gt_t_norm = gt_t / (gt_t.norm(dim=-1, keepdim=True).clamp(min=1e-6))
            losses["translation"] = self.weight_translation * F.l1_loss(pred_t_norm, gt_t_norm)

        losses["total"] = sum(losses.values())
        return losses

    def _geodesic_loss(self, pred_R, gt_R):
        R_diff = pred_R.transpose(-1, -2) @ gt_R
        trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos_angle = ((trace - 1) / 2).clamp(-1.0, 1.0)
        angle = torch.acos(cos_angle)
        return angle.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss for DA3 LoRA fine-tuning.

    Combines depth supervision and pose supervision with configurable weights.
    """

    def __init__(
        self,
        weight_depth=1.0,
        weight_pose=0.5,
        depth_kwargs=None,
        pose_kwargs=None,
    ):
        super().__init__()
        self.weight_depth = weight_depth
        self.weight_pose = weight_pose

        if depth_kwargs is None:
            depth_kwargs = {}
        if pose_kwargs is None:
            pose_kwargs = {}

        self.depth_loss = DepthLoss(**depth_kwargs)
        self.pose_loss = PoseLoss(**pose_kwargs)

    def forward(self, model_output, batch):
        losses = {}

        if "depths" in batch and "depth" in model_output:
            pred_depth = model_output["depth"]
            gt_depth = batch["depths"]
            images = batch["images"]

            if pred_depth.shape != gt_depth.shape:
                gt_depth = F.interpolate(
                    gt_depth.flatten(0, 1),
                    size=pred_depth.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).view(gt_depth.shape[0], gt_depth.shape[1], *pred_depth.shape[-2:])

            depth_losses = self.depth_loss(
                pred_depth,
                gt_depth,
                image=images.flatten(0, 1) if images is not None else None,
            )
            for k, v in depth_losses.items():
                losses["depth_{}".format(k)] = v * self.weight_depth

        if "extrinsics" in batch and "extrinsics" in model_output:
            pred_ext = model_output["extrinsics"]
            gt_ext = batch["extrinsics"]

            pose_losses = self.pose_loss(pred_ext, gt_ext)
            for k, v in pose_losses.items():
                losses["pose_{}".format(k)] = v * self.weight_pose

        losses["total"] = sum(v for k, v in losses.items() if k != "total")
        return losses
