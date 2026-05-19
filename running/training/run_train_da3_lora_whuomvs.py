from __future__ import annotations

import argparse
import json
import logging
import math
import os
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
from running.training.da3_metric_loss import MetricDepthLossV3

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


class DA3ForMetricDepth(nn.Module):
    def __init__(self, da3_model, scale_shift: MetricScaleShift):
        super().__init__()
        self.da3 = da3_model
        self.scale_shift = scale_shift

    def forward(self, images_input, extrinsics=None, intrinsics=None):
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            if extrinsics is not None and intrinsics is not None:
                output = self.da3.model(images_input, extrinsics=extrinsics, intrinsics=intrinsics)
            else:
                output = self.da3.model(images_input)
            pred_depth = output.depth[:, 0]
            if pred_depth.ndim == 3 and pred_depth.shape[1] == 1:
                pred_depth = pred_depth.squeeze(1)
            delta_log_scale = getattr(output, "metric_log_scale", None)
            delta_shift = getattr(output, "metric_shift", None)
        pred_depth = pred_depth.float()
        if delta_log_scale is not None and delta_log_scale.ndim > 1:
            delta_log_scale = delta_log_scale.squeeze(-1)
        if delta_shift is not None and delta_shift.ndim > 1:
            delta_shift = delta_shift.squeeze(-1)
        if delta_log_scale is None:
            delta_log_scale = torch.zeros(pred_depth.shape[0], device=pred_depth.device)
        if delta_shift is None:
            delta_shift = torch.zeros(pred_depth.shape[0], device=pred_depth.device)

        log_scale = self.scale_shift.log_scale + delta_log_scale
        shift = self.scale_shift.shift + delta_shift
        pred_metric = pred_depth * torch.exp(log_scale).view(-1, 1, 1) + shift.view(-1, 1, 1)
        return pred_depth, pred_metric, log_scale, shift


def get_raw_model(model):
    while isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module
    return model


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
                    pred = output.depth[:, 0]
                    if pred.ndim == 3 and pred.shape[1] == 1:
                        pred = pred.squeeze(1)
                pred = pred.float().cpu().numpy()[0]
            if pred.shape != depth_gt.shape:
                from PIL import Image as PILImage
                h, w = depth_gt.shape
                pred = np.array(PILImage.fromarray(pred).resize((w, h), PILImage.BILINEAR))
            valid = valid_mask & np.isfinite(pred) & np.isfinite(depth_gt) & (depth_gt > 1.0) & (pred > 1e-6)
            if valid.sum() < 100:
                continue
            pred_med = np.median(pred[valid])
            gt_med = np.median(depth_gt[valid])
            if np.isfinite(pred_med) and np.isfinite(gt_med) and pred_med > 1e-6:
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
    num_batches = 0
    num_valid_samples = 0
    logged_depth_images = False

    for batch in dataloader:
        images = batch["image"].to(device)
        depth_gt = batch["depth_gt"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        images_input = images.unsqueeze(1)

        _, pred_metric, log_scale, shift = raw(images_input)

        if pred_metric.shape[-2:] != depth_gt.shape[-2:]:
            pred_metric = F.interpolate(
                pred_metric.unsqueeze(1), size=depth_gt.shape[-2:],
                mode="bilinear", align_corners=True,
            ).squeeze(1)

        loss_dict = criterion(pred_metric, depth_gt, valid_mask, log_scale=log_scale, shift=shift)
        total_loss += loss_dict["loss"].item()

        if not logged_depth_images and writer is not None:
            logged_depth_images = True
            n_vis = min(images.shape[0], 2)
            for vi in range(n_vis):
                gt_vis = depth_gt[vi].cpu().numpy()
                pred_vis = pred_metric[vi].cpu().numpy()
                mask_vis = valid_mask[vi].cpu().numpy().astype(bool)
                gt_valid = gt_vis[mask_vis]

                def _norm(d, m):
                    v = d[m]
                    mx = np.abs(v).max() if v.size > 0 else 1.0
                    return d / max(mx, 1e-6)

                writer.add_image(
                    f"{tag}_depth/gt_{vi}",
                    torch.from_numpy(_norm(gt_vis, mask_vis)).unsqueeze(0),
                    global_step,
                )
                writer.add_image(
                    f"{tag}_depth/pred_{vi}",
                    torch.from_numpy(_norm(pred_vis, mask_vis)).unsqueeze(0),
                    global_step,
                )
                err_map = np.abs(pred_vis - gt_vis)
                gt_max = np.abs(gt_valid).max() if gt_valid.size > 0 else 1.0
                writer.add_image(
                    f"{tag}_depth/error_{vi}",
                    torch.from_numpy(err_map / max(gt_max, 1e-6)).unsqueeze(0),
                    global_step,
                )

        pred_np = pred_metric.cpu().numpy()
        gt_np = depth_gt.cpu().numpy()
        mask_np = valid_mask.cpu().numpy().astype(bool)
        for i in range(pred_np.shape[0]):
            m = mask_np[i] & np.isfinite(pred_np[i]) & np.isfinite(gt_np[i]) & (gt_np[i] > 1.0)
            if m.sum() < 10:
                continue
            p = pred_np[i][m]
            g = gt_np[i][m]
            total_abs_rel += np.mean(np.abs(p - g) / g)
            total_rmse += np.sqrt(np.mean((p - g) ** 2))
            total_mae += np.mean(np.abs(p - g))
            num_valid_samples += 1
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    n_valid = max(num_valid_samples, 1)
    metrics = {
        f"{tag}_loss": avg_loss,
        f"{tag}_abs_rel": total_abs_rel / n_valid,
        f"{tag}_rmse": total_rmse / n_valid,
        f"{tag}_mae": total_mae / n_valid,
        f"{tag}_scale": raw.scale_shift.scale,
        f"{tag}_shift": raw.scale_shift.shift_val,
    }
    if writer is not None:
        for k, v in metrics.items():
            writer.add_scalar(k, v, global_step)
    return metrics


def set_phase_grads(model, phase):
    raw = get_raw_model(model)
    if phase == 1:
        for param in raw.da3.parameters():
            param.requires_grad = False
        for param in raw.scale_shift.parameters():
            param.requires_grad = True
        if hasattr(raw.da3.model.head, "metric_adapter"):
            for param in raw.da3.model.head.metric_adapter.parameters():
                param.requires_grad = True
    elif phase == 2:
        for name, param in raw.da3.named_parameters():
            param.requires_grad = "lora" in name.lower() or "metric_adapter" in name.lower()
        for param in raw.scale_shift.parameters():
            param.requires_grad = True
    elif phase == 3:
        for name, param in raw.da3.named_parameters():
            param.requires_grad = (
                "lora" in name.lower()
                or "metric_adapter" in name.lower()
                or "output_conv" in name.lower()
            )
        for param in raw.scale_shift.parameters():
            param.requires_grad = True


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DA3-Large with LoRA + MetricAdapterV3 for metric depth on WHU-OMVS"
    )
    parser.add_argument("--dataset_root", type=str, default="dataset/WHU-OMVS")
    parser.add_argument("--output_dir", type=str, default="exp/da3_large_lora_whuomvs")
    parser.add_argument("--model_name", type=str, default="da3-large",
                        choices=["da3-base", "da3-large", "da3-giant", "da3-small"])
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="+", default=["qkv", "proj"])
    parser.add_argument("--adapter_hidden_dim", type=int, default=64)
    parser.add_argument("--adapter_depth_norm", type=float, default=600.0)
    parser.add_argument("--phase1_epochs", type=int, default=2)
    parser.add_argument("--phase2_epochs", type=int, default=10)
    parser.add_argument("--si_weight", type=float, default=1.0)
    parser.add_argument("--logl1_weight", type=float, default=10.0)
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--absrel_weight", type=float, default=0.5)
    parser.add_argument("--gradient_weight", type=float, default=0.5)
    parser.add_argument("--range_weight", type=float, default=0.1)
    parser.add_argument("--scale_reg_weight", type=float, default=0.01)
    parser.add_argument("--shift_reg_weight", type=float, default=0.01)
    parser.add_argument("--depth_norm", type=float, default=600.0)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_val_samples", type=int, default=-1)
    parser.add_argument("--max_test_samples", type=int, default=-1)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--save_every_n_epochs", type=int, default=5)
    parser.add_argument("--val_interval_steps", type=int, default=500)
    parser.add_argument("--print_every_steps", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    if args.adapter_depth_norm is not None:
        args.depth_norm = args.adapter_depth_norm

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
    if hasattr(da3_model.model, "head") and hasattr(da3_model.model.head, "metric_adapter"):
        from da3.model.metric_adapter import MetricAdapterV3

        feat_dim = da3_model.model.head.metric_adapter.residual_conv1.in_channels - 1
        da3_model.model.head.metric_adapter = MetricAdapterV3(
            feat_dim=feat_dim,
            hidden_dim=args.adapter_hidden_dim,
            depth_norm=args.depth_norm,
        ).to(device)

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

    init_scale = init_scale_from_dataset(da3_model, train_dataset, device, max_samples=200)
    scale_shift = MetricScaleShift(init_scale=init_scale, init_shift=0.0).to(device)

    model = DA3ForMetricDepth(da3_model, scale_shift)

    logger.info("Applying LoRA: rank=%d, alpha=%d, targets=%s",
                args.lora_rank, args.lora_alpha, args.lora_target_modules)
    da3_inner = da3_model.model
    da3_inner = apply_lora_to_model(
        da3_inner, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=args.lora_target_modules,
    )
    da3_model.model = da3_inner

    raw_model = get_raw_model(model)
    if hasattr(raw_model.da3.model.head, "metric_adapter"):
        head_params = sum(p.numel() for p in raw_model.da3.model.head.metric_adapter.parameters())
    else:
        head_params = 0
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Trainable: %d / %d (%.2f%%) | MetricAdapter: %d | ScaleShift: %d",
                trainable_params, total_params, 100.0 * trainable_params / total_params,
                head_params, sum(p.numel() for p in raw_model.scale_shift.parameters()))
    logger.info("ScaleShift: scale=%.4f, shift=%.4f", raw_model.scale_shift.scale, raw_model.scale_shift.shift_val)

    criterion = MetricDepthLossV3(
        si_weight=args.si_weight, logl1_weight=args.logl1_weight,
        l1_weight=args.l1_weight, absrel_weight=args.absrel_weight,
        gradient_weight=args.gradient_weight, range_weight=args.range_weight,
        scale_reg_weight=args.scale_reg_weight, shift_reg_weight=args.shift_reg_weight,
        depth_norm=args.depth_norm,
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
    warmup_steps = min(args.warmup_steps, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    best_val_mae = float("inf")
    best_test_mae = float("inf")
    global_step = 0

    if args.resume and os.path.exists(args.resume):
        logger.info("Resuming from checkpoint: %s", args.resume)
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "global_step" in ckpt:
            global_step = ckpt["global_step"]
        if "best_val_mae" in ckpt:
            best_val_mae = ckpt["best_val_mae"]
        logger.info("Resumed from global_step=%d, best_val_mae=%.4f", global_step, best_val_mae)

    logger.info("=" * 70)
    logger.info("Training config:")
    logger.info("  Model: %s (%.1fM params)", args.model_name, total_params / 1e6)
    logger.info("  Train: %d samples, Val: %d samples, Test: %d samples",
                len(train_dataset), len(val_dataset), len(test_dataset))
    logger.info("  Epochs: %d (Phase1=%d, Phase2=%d, Phase3=%d)",
                args.epochs, args.phase1_epochs, args.phase2_epochs,
                args.epochs - args.phase1_epochs - args.phase2_epochs)
    logger.info("  LR: %.2e, Warmup: %d steps, Grad accum: %d",
                args.lr, warmup_steps, args.grad_accum_steps)
    logger.info("  LoRA: rank=%d, alpha=%d", args.lora_rank, args.lora_alpha)
    logger.info("  Loss: SI=%.1f LogL1=%.1f L1=%.1f AbsRel=%.1f Grad=%.1f Range=%.1f",
                args.si_weight, args.logl1_weight, args.l1_weight,
                args.absrel_weight, args.gradient_weight, args.range_weight)
    logger.info("  Reg: scale=%.4f shift=%.4f depth_norm=%.1f",
                args.scale_reg_weight, args.shift_reg_weight, args.depth_norm)
    logger.info("  Validation every %d steps", args.val_interval_steps)
    logger.info("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        if epoch < args.phase1_epochs:
            phase = 1
        elif epoch < args.phase1_epochs + args.phase2_epochs:
            phase = 2
        else:
            phase = 3

        set_phase_grads(model, phase)

        trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
        current_lr = scheduler.get_last_lr()[0]
        logger.info("=" * 70)
        logger.info("Epoch %d/%d Phase%d | trainable=%d lr=%.2e scale=%.4f shift=%.4f",
                     epoch + 1, args.epochs, phase, trainable_now, current_lr,
                     raw_model.scale_shift.scale, raw_model.scale_shift.shift_val)
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
            images_input = images.unsqueeze(1)

            _, pred_metric, log_scale, shift = model(images_input)

            if pred_metric.shape[-2:] != depth_gt.shape[-2:]:
                pred_metric = F.interpolate(
                    pred_metric.unsqueeze(1), size=depth_gt.shape[-2:],
                    mode="bilinear", align_corners=True,
                ).squeeze(1)

            loss_dict = criterion(pred_metric, depth_gt, valid_mask, log_scale=log_scale, shift=shift)
            loss = loss_dict["loss"] / args.grad_accum_steps
            loss.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar("train/loss_total", loss_dict["loss"].item(), global_step)
                writer.add_scalar("train/si_loss", loss_dict["si_loss"].item(), global_step)
                writer.add_scalar("train/logl1_loss", loss_dict["logl1_loss"].item(), global_step)
                writer.add_scalar("train/l1_loss", loss_dict["l1_loss"].item(), global_step)
                writer.add_scalar("train/absrel_loss", loss_dict["absrel_loss"].item(), global_step)
                writer.add_scalar("train/gradient_loss", loss_dict["gradient_loss"].item(), global_step)
                writer.add_scalar("train/range_loss", loss_dict["range_loss"].item(), global_step)
                writer.add_scalar("train/scale_reg", loss_dict["scale_reg"].item(), global_step)
                writer.add_scalar("train/shift_reg", loss_dict["shift_reg"].item(), global_step)
                writer.add_scalar("train/scale", raw_model.scale_shift.scale, global_step)
                writer.add_scalar("train/shift", raw_model.scale_shift.shift_val, global_step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)
                writer.add_scalar("train/lr", current_lr, global_step)
                writer.add_scalar("train/epoch", epoch + 1, global_step)

            epoch_loss += loss_dict["loss"].item()
            epoch_l1 += loss_dict["l1_loss"].item()
            num_batches += 1

            if global_step > 0 and (global_step % args.print_every_steps == 0):
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    "  E%d/%d P%d GStep %d | lr=%.2e loss=%.2f si=%.4f logl1=%.4f l1=%.2f absrel=%.4f scale=%.2f shift=%.4f",
                    epoch + 1, args.epochs, phase, global_step,
                    current_lr, loss_dict["loss"].item(), loss_dict["si_loss"].item(),
                    loss_dict["logl1_loss"].item(), loss_dict["l1_loss"].item(),
                    loss_dict["absrel_loss"].item(),
                    raw_model.scale_shift.scale, raw_model.scale_shift.shift_val,
                )

            if args.val_interval_steps > 0 and (step + 1) % args.val_interval_steps == 0:
                val_metrics = validate(model, val_loader, criterion, device, global_step, writer, tag="val")
                test_metrics = validate(model, test_loader, criterion, device, global_step, writer, tag="test")
                logger.info(
                    "  [Val] E%d Step %d | mae=%.2f rmse=%.2f abs_rel=%.4f scale=%.2f shift=%.4f",
                    epoch + 1, step,
                    val_metrics["val_mae"], val_metrics["val_rmse"],
                    val_metrics["val_abs_rel"],
                    val_metrics["val_scale"], val_metrics["val_shift"],
                )
                logger.info(
                    "  [Test] E%d Step %d | mae=%.2f rmse=%.2f abs_rel=%.4f",
                    epoch + 1, step,
                    test_metrics["test_mae"], test_metrics["test_rmse"],
                    test_metrics["test_abs_rel"],
                )
                is_best = val_metrics["val_mae"] < best_val_mae
                if is_best:
                    best_val_mae = val_metrics["val_mae"]
                    best_test_mae = test_metrics["test_mae"]
                    best_path = ckpt_dir / "best.pt"
                    torch.save({
                        "epoch": epoch + 1, "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "val_mae": val_metrics["val_mae"],
                        "test_mae": test_metrics["test_mae"],
                        "best_val_mae": best_val_mae,
                        "best_test_mae": best_test_mae,
                        "val_metrics": val_metrics,
                        "test_metrics": test_metrics,
                        "scale": raw_model.scale_shift.scale,
                        "shift_val": raw_model.scale_shift.shift_val,
                        "args": vars(args),
                    }, str(best_path))
                    logger.info("  *** New best! val_mae=%.4f test_mae=%.4f ***",
                                val_metrics["val_mae"], test_metrics["test_mae"])
                model.train()
                set_phase_grads(model, phase)

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        avg_epoch_l1 = epoch_l1 / max(num_batches, 1)
        logger.info("Epoch %d done | avg_loss=%.4f avg_l1=%.2f scale=%.4f shift=%.4f",
                     epoch + 1, avg_epoch_loss, avg_epoch_l1,
                     raw_model.scale_shift.scale, raw_model.scale_shift.shift_val)
        writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)
        writer.add_scalar("train/epoch_l1", avg_epoch_l1, epoch)

        val_metrics = validate(model, val_loader, criterion, device, global_step, writer, tag="val")
        test_metrics = validate(model, test_loader, criterion, device, global_step, writer, tag="test")
        logger.info(
            "  [EpochVal] E%d | val_mae=%.2f val_rmse=%.2f test_mae=%.2f test_rmse=%.2f scale=%.2f shift=%.4f",
            epoch + 1,
            val_metrics["val_mae"], val_metrics["val_rmse"],
            test_metrics["test_mae"], test_metrics["test_rmse"],
            val_metrics["val_scale"], val_metrics["val_shift"],
        )

        is_best = val_metrics["val_mae"] < best_val_mae
        if is_best:
            best_val_mae = val_metrics["val_mae"]
            best_test_mae = test_metrics["test_mae"]

        if (epoch + 1) % args.save_every_n_epochs == 0 or is_best:
            ckpt_path = ckpt_dir / f"epoch_{epoch+1:03d}.pt"
            torch.save({
                "epoch": epoch + 1, "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_mae": val_metrics["val_mae"],
                "test_mae": test_metrics["test_mae"],
                "best_val_mae": best_val_mae,
                "best_test_mae": best_test_mae,
                "scale": raw_model.scale_shift.scale,
                "shift_val": raw_model.scale_shift.shift_val,
                "args": vars(args),
            }, str(ckpt_path))
            logger.info("Saved checkpoint: %s (best=%s, val_mae=%.4f, test_mae=%.4f)",
                        ckpt_path, is_best, val_metrics["val_mae"], test_metrics["test_mae"])

            if is_best:
                best_path = ckpt_dir / "best.pt"
                torch.save({
                    "epoch": epoch + 1, "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "val_mae": val_metrics["val_mae"],
                    "test_mae": test_metrics["test_mae"],
                    "best_val_mae": best_val_mae,
                    "best_test_mae": best_test_mae,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "scale": raw_model.scale_shift.scale,
                    "shift_val": raw_model.scale_shift.shift_val,
                    "args": vars(args),
                }, str(best_path))
                logger.info("New best model! val_mae=%.4f test_mae=%.4f",
                            val_metrics["val_mae"], test_metrics["test_mae"])

    scale_shift_path = ckpt_dir / "scale_shift.pt"
    torch.save({
        "log_scale": raw_model.scale_shift.log_scale.data,
        "shift": raw_model.scale_shift.shift.data,
        "scale": raw_model.scale_shift.scale,
        "shift_val": raw_model.scale_shift.shift_val,
    }, str(scale_shift_path))
    logger.info("Saved ScaleShift: %s (scale=%.4f, shift=%.4f)",
                scale_shift_path, raw_model.scale_shift.scale, raw_model.scale_shift.shift_val)

    lora_path = ckpt_dir / "lora_final.pt"
    lora_state = {
        k: v for k, v in model.state_dict().items()
        if "lora" in k.lower() or "metric_adapter" in k.lower() or "scale_shift" in k.lower()
    }
    torch.save(lora_state, str(lora_path))
    logger.info("Saved LoRA + Head + ScaleShift weights: %s (%d keys)", lora_path, len(lora_state))

    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "best_val_mae": best_val_mae,
            "best_test_mae": best_test_mae,
            "final_scale": raw_model.scale_shift.scale,
            "final_shift": raw_model.scale_shift.shift_val,
            "args": vars(args),
        }, f, indent=2)

    writer.close()
    logger.info("Training complete. Best val_mae=%.4f, best_test_mae=%.4f", best_val_mae, best_test_mae)


if __name__ == "__main__":
    main()
