import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR
while _REPO_ROOT != _REPO_ROOT.parent and not (_REPO_ROOT / "models").exists():
    _REPO_ROOT = _REPO_ROOT.parent
_DA3_SRC = _REPO_ROOT / "models" / "da3" / "src"
for p in (_REPO_ROOT, _SCRIPT_DIR.parent, _DA3_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from running.data.whuomvs_depth_dataset import WHUOMVSDepthDataset

logger = logging.getLogger(__name__)


class MetricDepthLoss(nn.Module):
    def __init__(
        self,
        si_weight: float = 1.0,
        l1_weight: float = 1.0,
        affine_weight: float = 0.5,
        gradient_weight: float = 0.1,
        depth_range: Tuple[float, float] = (1.0, 2000.0),
    ):
        super().__init__()
        self.si_weight = si_weight
        self.l1_weight = l1_weight
        self.affine_weight = affine_weight
        self.gradient_weight = gradient_weight
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
                "l1_loss": zero.clone(),
                "affine_loss": zero.clone(),
                "gradient_loss": zero.clone(),
                "scale": torch.tensor(1.0, device=pred_depth.device),
                "shift": torch.tensor(0.0, device=pred_depth.device),
            }

        pred_valid = pred_depth[valid_mask]
        gt_valid = gt_depth[valid_mask]

        scale, shift = self._compute_affine_params(pred_valid, gt_valid)
        pred_aligned = pred_depth * scale + shift

        si_loss = self._scale_invariant_log_loss(pred_aligned, gt_depth, valid_mask)
        l1_loss = F.l1_loss(pred_aligned[valid_mask], gt_valid)
        affine_loss = self._affine_regularization(scale, shift)
        gradient_loss = self._gradient_loss(pred_aligned, gt_depth, valid_mask)

        total_loss = (
            self.si_weight * si_loss
            + self.l1_weight * l1_loss
            + self.affine_weight * affine_loss
            + self.gradient_weight * gradient_loss
        )

        return {
            "loss": total_loss,
            "si_loss": si_loss.detach(),
            "l1_loss": l1_loss.detach(),
            "affine_loss": affine_loss.detach(),
            "gradient_loss": gradient_loss.detach(),
            "scale": scale.detach(),
            "shift": shift.detach(),
        }

    def _compute_affine_params(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = pred.numel()
        max_samples = 50000
        if n > max_samples:
            idx = torch.randperm(n, device=pred.device)[:max_samples]
            pred_s = pred.reshape(-1)[idx]
            gt_s = gt.reshape(-1)[idx]
        else:
            pred_s = pred.reshape(-1)
            gt_s = gt.reshape(-1)

        n_s = pred_s.numel()
        s_xy = (pred_s * gt_s).sum()
        s_xx = (pred_s * pred_s).sum()
        s_x = pred_s.sum()
        s_y = gt_s.sum()

        det = n_s * s_xx - s_x * s_x
        if torch.abs(det) < 1e-8:
            return torch.tensor(1.0, device=pred.device), torch.tensor(0.0, device=pred.device)

        scale = (n_s * s_xy - s_x * s_y) / det
        shift = (s_y * s_xx - s_x * s_xy) / det

        if not torch.isfinite(scale) or torch.abs(scale) < 1e-8:
            scale = torch.tensor(1.0, device=pred.device)
            shift = torch.tensor(0.0, device=pred.device)
        return scale, shift

    def _scale_invariant_log_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        log_pred = torch.log(pred[valid_mask].clamp(min=1e-6))
        log_gt = torch.log(gt[valid_mask].clamp(min=1e-6))
        diff = log_pred - log_gt
        si_loss = (diff ** 2).mean() - 0.5 * (diff.mean()) ** 2
        return si_loss

    def _affine_regularization(
        self, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        log_scale = torch.log(torch.abs(scale) + 1e-6)
        scale_reg = log_scale ** 2
        shift_reg = torch.log1p(torch.abs(shift)) ** 2
        return scale_reg + shift_reg

    def _gradient_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if pred.ndim != 2 or gt.ndim != 2:
            return torch.tensor(0.0, device=pred.device)

        pred_dx = pred[:, 1:] - pred[:, :-1]
        pred_dy = pred[1:, :] - pred[:-1, :]
        gt_dx = gt[:, 1:] - gt[:, :-1]
        gt_dy = gt[1:, :] - gt[:-1, :]

        mask_dx = valid_mask[:, 1:] & valid_mask[:, :-1]
        mask_dy = valid_mask[1:, :] & valid_mask[:-1, :]

        n_dx = mask_dx.sum().clamp(min=1)
        n_dy = mask_dy.sum().clamp(min=1)

        loss_dx = (torch.where(mask_dx, (pred_dx - gt_dx) ** 2, torch.zeros_like(pred_dx)).sum()) / n_dx
        loss_dy = (torch.where(mask_dy, (pred_dy - gt_dy) ** 2, torch.zeros_like(pred_dy)).sum()) / n_dy
        return loss_dx + loss_dy


def apply_lora_to_model(
    model: nn.Module,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
) -> nn.Module:
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


def freeze_backbone_unfreeze_head(model: nn.Module, head_name: str = "head"):
    for name, param in model.named_parameters():
        if head_name in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def build_da3_model(model_name: str = "da3-base", pretrained_path: Optional[str] = None):
    from models.da3.api import DepthAnything3

    model = DepthAnything3(model_name=model_name)
    if pretrained_path and os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded pretrained weights from {pretrained_path}")
    return model


def build_w2c_extrinsic(Rwc: torch.Tensor, twc: torch.Tensor) -> torch.Tensor:
    O_xrightyup = torch.tensor(
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float64, device=Rwc.device
    )
    Rwc_down = Rwc.double() @ O_xrightyup
    c2w = torch.eye(4, dtype=torch.float64, device=Rwc.device)
    c2w[:3, :3] = Rwc_down
    c2w[:3, 3] = twc.double()
    w2c = torch.linalg.inv(c2w)
    return w2c.float()


def build_intrinsic_3x3(K: torch.Tensor) -> torch.Tensor:
    return K


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    criterion,
    device,
    epoch,
    writer,
    grad_accum_steps=4,
    use_multiview=False,
):
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        depth_gt = batch["depth_gt"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        B = images.shape[0]

        if use_multiview:
            Rwc = batch["Rwc"].to(device)
            twc = batch["twc"].to(device)
            K = batch["K"].to(device)

            extrinsics_list = []
            intrinsics_list = []
            for i in range(B):
                w2c = build_w2c_extrinsic(Rwc[i], twc[i])
                extrinsics_list.append(w2c)
                intrinsics_list.append(K[i])

            extrinsics = torch.stack(extrinsics_list, dim=0).unsqueeze(1)
            intrinsics = torch.stack(intrinsics_list, dim=0).unsqueeze(1)

            images_input = images.unsqueeze(1)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                output = model.model(
                    images_input,
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                )
                pred_depth = output.depth[:, 0]
                if pred_depth.ndim == 3 and pred_depth.shape[1] == 1:
                    pred_depth = pred_depth.squeeze(1)
        else:
            images_input = images.unsqueeze(1)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                output = model.model(images_input)
                pred_depth = output.depth[:, 0]
                if pred_depth.ndim == 3 and pred_depth.shape[1] == 1:
                    pred_depth = pred_depth.squeeze(1)

        if pred_depth.shape[-2:] != depth_gt.shape[-2:]:
            pred_depth = F.interpolate(
                pred_depth.unsqueeze(1),
                size=depth_gt.shape[-2:],
                mode="bilinear",
                align_corners=True,
            ).squeeze(1)

        loss_dict = criterion(pred_depth, depth_gt, valid_mask)
        loss = loss_dict["loss"] / grad_accum_steps

        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step = epoch * len(dataloader) + step
            writer.add_scalar("train/loss_total", loss_dict["loss"].item(), global_step)
            writer.add_scalar("train/si_loss", loss_dict["si_loss"].item(), global_step)
            writer.add_scalar("train/l1_loss", loss_dict["l1_loss"].item(), global_step)
            writer.add_scalar("train/affine_loss", loss_dict["affine_loss"].item(), global_step)
            writer.add_scalar("train/gradient_loss", loss_dict["gradient_loss"].item(), global_step)
            writer.add_scalar("train/scale", loss_dict["scale"].item(), global_step)
            writer.add_scalar("train/shift", loss_dict["shift"].item(), global_step)
            writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

        total_loss += loss_dict["loss"].item()
        num_batches += 1

        if step % 20 == 0:
            logger.info(
                f"  Epoch {epoch} Step {step}/{len(dataloader)} | "
                f"loss={loss_dict['loss'].item():.4f} "
                f"si={loss_dict['si_loss'].item():.4f} "
                f"l1={loss_dict['l1_loss'].item():.4f} "
                f"affine={loss_dict['affine_loss'].item():.4f} "
                f"scale={loss_dict['scale'].item():.4f} "
                f"shift={loss_dict['shift'].item():.4f}"
            )

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0.0
    total_abs_rel = 0.0
    total_sq_rel = 0.0
    total_rmse = 0.0
    total_scale = 0.0
    total_shift = 0.0
    num_batches = 0
    num_valid_samples = 0
    logged_depth_images = False

    for step, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        depth_gt = batch["depth_gt"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        images_input = images.unsqueeze(1)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            output = model.model(images_input)
            pred_depth = output.depth[:, 0]
            if pred_depth.ndim == 3 and pred_depth.shape[1] == 1:
                pred_depth = pred_depth.squeeze(1)

        if pred_depth.shape[-2:] != depth_gt.shape[-2:]:
            pred_depth = F.interpolate(
                pred_depth.unsqueeze(1),
                size=depth_gt.shape[-2:],
                mode="bilinear",
                align_corners=True,
            ).squeeze(1)

        loss_dict = criterion(pred_depth, depth_gt, valid_mask)
        total_loss += loss_dict["loss"].item()
        total_scale += loss_dict["scale"].item()
        total_shift += loss_dict["shift"].item()

        if not logged_depth_images and writer is not None:
            logged_depth_images = True
            n_vis = min(images.shape[0], 2)
            for vi in range(n_vis):
                gt_vis = depth_gt[vi].cpu().numpy()
                pred_vis = pred_depth[vi].cpu().numpy()
                mask_vis = valid_mask[vi].cpu().numpy().astype(bool)

                gt_valid = gt_vis[mask_vis]
                pred_valid = pred_vis[mask_vis]
                if gt_valid.max() > 0:
                    gt_norm = gt_vis / max(gt_valid.max(), 1e-6)
                else:
                    gt_norm = gt_vis
                if pred_valid.max() > 0:
                    pred_norm = pred_vis / max(pred_valid.max(), 1e-6)
                else:
                    pred_norm = pred_vis

                gt_tensor = torch.from_numpy(gt_norm).unsqueeze(0)
                pred_tensor = torch.from_numpy(pred_norm).unsqueeze(0)
                writer.add_image(f"val_depth/gt_{vi}", gt_tensor, epoch)
                writer.add_image(f"val_depth/pred_{vi}", pred_tensor, epoch)

                err_map = np.abs(pred_vis - gt_vis)
                if gt_valid.max() > 0:
                    err_norm = err_map / max(gt_valid.max(), 1e-6)
                else:
                    err_norm = err_map
                err_tensor = torch.from_numpy(err_norm).unsqueeze(0)
                writer.add_image(f"val_depth/error_{vi}", err_tensor, epoch)

        pred_np = pred_depth.cpu().numpy()
        gt_np = depth_gt.cpu().numpy()
        mask_np = valid_mask.cpu().numpy().astype(bool)

        for i in range(pred_np.shape[0]):
            m = mask_np[i] & np.isfinite(pred_np[i]) & np.isfinite(gt_np[i]) & (gt_np[i] > 1.0)
            if m.sum() < 10:
                continue
            p = pred_np[i][m]
            g = gt_np[i][m]
            scale = np.median(g) / max(np.median(p), 1e-6)
            p_aligned = p * scale
            abs_rel = np.mean(np.abs(p_aligned - g) / g)
            sq_rel = np.mean(((p_aligned - g) ** 2) / g)
            rmse = np.sqrt(np.mean((p_aligned - g) ** 2))
            total_abs_rel += abs_rel
            total_sq_rel += sq_rel
            total_rmse += rmse
            num_valid_samples += 1

        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_scale = total_scale / max(num_batches, 1)
    avg_shift = total_shift / max(num_batches, 1)
    n_valid = max(num_valid_samples, 1)
    metrics = {
        "val_loss": avg_loss,
        "abs_rel": total_abs_rel / n_valid,
        "sq_rel": total_sq_rel / n_valid,
        "rmse": total_rmse / n_valid,
        "val_scale": avg_scale,
        "val_shift": avg_shift,
    }

    for k, v in metrics.items():
        writer.add_scalar(f"val/{k}", v, epoch)

    logger.info(
        f"  Validation Epoch {epoch} | "
        f"loss={metrics['val_loss']:.4f} "
        f"abs_rel={metrics['abs_rel']:.4f} "
        f"rmse={metrics['rmse']:.4f} "
        f"scale={metrics['val_scale']:.4f} "
        f"shift={metrics['val_shift']:.4f}"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DA3 with LoRA for metric depth on WHU-OMVS")
    parser.add_argument("--dataset_root", type=str, default="dataset/WHU-OMVS")
    parser.add_argument("--output_dir", type=str, default="exp/da3_base_lora_whuomvs")
    parser.add_argument("--model_name", type=str, default="da3-base", choices=["da3-base", "da3-large", "da3-giant", "da3-small"])
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="+", default=["qkv", "proj"])
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze_head", action="store_true", default=True)
    parser.add_argument("--use_multiview", action="store_true", default=False)
    parser.add_argument("--si_weight", type=float, default=1.0)
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--affine_weight", type=float, default=1.0)
    parser.add_argument("--gradient_weight", type=float, default=0.1)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_val_samples", type=int, default=-1)
    parser.add_argument("--val_split_ratio", type=float, default=0.1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--save_every_n_epochs", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    writer = SummaryWriter(log_dir=str(output_dir / "logs"))

    logger.info("Building DA3 model: %s", args.model_name)
    model = build_da3_model(args.model_name, args.pretrained_path)
    model = model.to(device)

    if args.freeze_backbone:
        logger.info("Freezing backbone, unfreezing head")
        freeze_backbone_unfreeze_head(model.model, head_name="head")

    logger.info("Applying LoRA: rank=%d, alpha=%d, targets=%s", args.lora_rank, args.lora_alpha, args.lora_target_modules)
    model_inner = model.model
    model_inner = apply_lora_to_model(
        model_inner,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )
    model.model = model_inner

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Trainable: %d / %d (%.2f%%)", trainable_params, total_params, 100.0 * trainable_params / total_params)

    criterion = MetricDepthLoss(
        si_weight=args.si_weight,
        l1_weight=args.l1_weight,
        affine_weight=args.affine_weight,
        gradient_weight=args.gradient_weight,
    )

    dataset_root = str(Path(args.dataset_root).resolve() if not os.path.isabs(args.dataset_root) else args.dataset_root)
    train_areas = ["area1", "area4", "area5", "area6"]

    train_dataset = WHUOMVSDepthDataset(
        dataset_root=dataset_root,
        split="train",
        areas=train_areas,
        process_res=args.process_res,
        augment=True,
        max_samples=args.max_train_samples if args.max_train_samples > 0 else -1,
    )

    val_dataset = WHUOMVSDepthDataset(
        dataset_root=dataset_root,
        split="train",
        areas=train_areas[:1],
        process_res=args.process_res,
        augment=False,
        max_samples=args.max_val_samples if args.max_val_samples > 0 else 500,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = max(args.epochs * len(train_loader) // args.grad_accum_steps, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.lr * 0.01,
    )

    logger.info("Starting training: %d epochs, %d steps/epoch, %d total steps", args.epochs, len(train_loader), total_steps)

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        logger.info("=== Epoch %d/%d ===", epoch + 1, args.epochs)
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            epoch=epoch,
            writer=writer,
            grad_accum_steps=args.grad_accum_steps,
            use_multiview=args.use_multiview,
        )
        logger.info("Epoch %d train_loss=%.4f", epoch + 1, train_loss)
        writer.add_scalar("train/epoch_loss", train_loss, epoch)

        val_metrics = validate(model, val_loader, criterion, device, epoch, writer)

        if (epoch + 1) % args.save_every_n_epochs == 0 or val_metrics["val_loss"] < best_val_loss:
            is_best = val_metrics["val_loss"] < best_val_loss
            best_val_loss = min(best_val_loss, val_metrics["val_loss"])

            ckpt_path = ckpt_dir / f"epoch_{epoch+1:03d}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_metrics["val_loss"],
                "args": vars(args),
            }, str(ckpt_path))
            logger.info("Saved checkpoint: %s (best=%s)", ckpt_path, is_best)

            if is_best:
                best_path = ckpt_dir / "best.pt"
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_metrics["val_loss"],
                    "val_metrics": val_metrics,
                    "args": vars(args),
                }, str(best_path))
                logger.info("New best model! val_loss=%.4f", val_metrics["val_loss"])

    lora_path = ckpt_dir / "lora_final.pt"
    lora_state = {k: v for k, v in model.state_dict().items() if "lora" in k.lower() or "head" in k.lower()}
    torch.save(lora_state, str(lora_path))
    logger.info("Saved LoRA weights: %s (%d params)", lora_path, len(lora_state))

    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"best_val_loss": best_val_loss, "args": vars(args)}, f, indent=2)

    writer.close()
    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)


if __name__ == "__main__":
    main()
