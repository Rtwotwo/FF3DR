"""
DA3 LoRA Fine-tuning on MatrixCity for Drone-View Depth Estimation.

Usage:
    python -m running.training.train_da3_lora --config configs/train_da3_lora.yaml
    bash running/scripts/train_da3_lora.sh
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "models"))

from da3lora.lora import apply_lora_to_model, freeze_non_lora_params, get_lora_params, save_lora_weights, load_lora_weights
from da3lora.cfg import load_config, create_object
from da3lora.registry import MODEL_REGISTRY
from running.training.datasets.matrixcity import MatrixCityDataset, collate_fn
from running.training.utils.losses import CombinedLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="DA3 LoRA Fine-tuning on MatrixCity")
    parser.add_argument("--config", type=str, default="", help="YAML config file path")
    parser.add_argument("--dataset_dir", type=str,
                        default=str(_REPO_ROOT / "dataset" / "MatrixCity"),
                        help="MatrixCity dataset root directory")
    parser.add_argument("--city_size", type=str, default="big_city",
                        choices=["big_city", "small_city"])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model_name", type=str, default="da3-large",
                        help="DA3 model preset name")
    parser.add_argument("--pretrained_path", type=str, default="",
                        help="Path to pretrained DA3 weights")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--num_views", type=int, default=2, help="Number of views per sample")
    parser.add_argument("--image_size", type=int, default=504, help="Training image size")
    parser.add_argument("--stride", type=int, default=10, help="Frame sampling stride")
    parser.add_argument("--max_depth", type=float, default=500.0, help="Max depth for normalization")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for LoRA params")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500, help="LR warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--depth_loss_weight", type=float, default=1.0)
    parser.add_argument("--pose_loss_weight", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="", help="Output directory")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--log_every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--resume", type=str, default="", help="Resume from LoRA checkpoint")
    parser.add_argument("--gpu_id", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth_supervision", type=int, default=1,
                        help="Use depth GT for supervision (1=yes, 0=no)")
    return parser.parse_args()


def build_model(args):
    logger.info("Building DA3 model: {}".format(args.model_name))
    config = load_config(MODEL_REGISTRY[args.model_name])
    model = create_object(config)

    if args.pretrained_path:
        logger.info("Loading pretrained weights from {}".format(args.pretrained_path))
        if os.path.isdir(args.pretrained_path):
            from da3lora.utils.model_loading import load_pretrained
            load_pretrained(model, args.pretrained_path)
        elif args.pretrained_path.endswith(".safetensors"):
            from da3lora.utils.model_loading import load_pretrained
            load_pretrained(model, args.pretrained_path)
        elif args.pretrained_path.endswith(".pt") or args.pretrained_path.endswith(".pth"):
            state_dict = torch.load(args.pretrained_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            logger.info("Loaded .pt/.pth checkpoint")

    logger.info("Applying LoRA: r={}, alpha={}, dropout={}".format(
        args.lora_r, args.lora_alpha, args.lora_dropout))
    model = apply_lora_to_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    freeze_non_lora_params(model)

    return model


def build_dataloader(args):
    dataset = MatrixCityDataset(
        dataset_dir=args.dataset_dir,
        city_size=args.city_size,
        split=args.split,
        num_views=args.num_views,
        max_depth=args.max_depth,
        image_size=args.image_size,
        stride=args.stride,
        depth_supervision=bool(args.depth_supervision),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader


def build_optimizer(model, args):
    lora_params = get_lora_params(model)
    param_groups = [
        {"params": lora_params["lora_A"], "lr": args.lr, "name": "lora_A"},
        {"params": lora_params["lora_B"], "lr": args.lr * 0.5, "name": "lora_B"},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
    )
    return optimizer


def build_scheduler(optimizer, args, total_steps):
    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(
            max(1, total_steps - args.warmup_steps)
        )
        return max(0.1, 1.0 - 0.9 * progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def train_one_epoch(
    model, dataloader, optimizer, scheduler, criterion, scaler, device, args, epoch, writer, global_step
):
    model.train()
    epoch_loss = 0.0
    epoch_depth_loss = 0.0
    epoch_pose_loss = 0.0
    num_steps = 0

    for batch_idx, batch in enumerate(dataloader):
        images = batch["images"].to(device)
        gt_extrinsics = batch["extrinsics"].to(device)
        gt_intrinsics = batch["intrinsics"].to(device)
        gt_depths = batch.get("depths")
        if gt_depths is not None:
            gt_depths = gt_depths.to(device)

        B, N, C, H, W = images.shape

        with autocast(enabled=True, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            output = model(
                images,
                extrinsics=gt_extrinsics,
                intrinsics=gt_intrinsics,
            )

            model_output = {}
            if hasattr(output, "depth"):
                model_output["depth"] = output.depth
            if hasattr(output, "extrinsics"):
                model_output["extrinsics"] = output.extrinsics
            if hasattr(output, "intrinsics"):
                model_output["intrinsics"] = output.intrinsics

            loss_batch = {"depths": gt_depths, "extrinsics": gt_extrinsics, "images": images}
            losses = criterion(model_output, loss_batch)
            loss = losses.get("total", torch.tensor(0.0, device=device))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], args.max_grad_norm
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        epoch_loss += loss.item()
        if "depth_total" in losses:
            epoch_depth_loss += losses["depth_total"].item()
        if "pose_total" in losses:
            epoch_pose_loss += losses["pose_total"].item()
        num_steps += 1
        global_step += 1

        if batch_idx % args.log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch [{}/{}] Step [{}/{}] Loss: {:.4f} LR: {:.6f}".format(
                    epoch, args.epochs, batch_idx, len(dataloader), loss.item(), lr)
            )
            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)
                if "depth_total" in losses:
                    writer.add_scalar("train/depth_loss", losses["depth_total"].item(), global_step)
                if "pose_total" in losses:
                    writer.add_scalar("train/pose_loss", losses["pose_total"].item(), global_step)

    avg_loss = epoch_loss / max(num_steps, 1)
    avg_depth = epoch_depth_loss / max(num_steps, 1)
    avg_pose = epoch_pose_loss / max(num_steps, 1)

    return avg_loss, avg_depth, avg_pose, global_step


def main():
    args = parse_args()

    if args.config and os.path.isfile(args.config):
        import yaml
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        train_cfg = cfg.get("Training", {})
        for k, v in train_cfg.items():
            if hasattr(args, k) and v is not None:
                setattr(args, k, type(getattr(args, k))(v))

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not args.output_dir:
        args.output_dir = str(
            _REPO_ROOT / "exp" / "da3lora_{}_r{}".format(args.city_size, args.lora_r)
        )
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("DA3 LoRA Fine-tuning on MatrixCity")
    logger.info("  Model: {}".format(args.model_name))
    logger.info("  LoRA: r={}, alpha={}".format(args.lora_r, args.lora_alpha))
    logger.info("  Dataset: {}/{}".format(args.city_size, args.split))
    logger.info("  Output: {}".format(args.output_dir))
    logger.info("  Device: {}".format(device))
    logger.info("=" * 60)

    model = build_model(args)
    model = model.to(device)

    if args.resume:
        logger.info("Resuming LoRA weights from {}".format(args.resume))
        load_lora_weights(model, args.resume)

    dataloader = build_dataloader(args)
    optimizer = build_optimizer(model, args)
    total_steps = len(dataloader) * args.epochs
    scheduler = build_scheduler(optimizer, args, total_steps)
    scaler = GradScaler(enabled=True)
    criterion = CombinedLoss(
        weight_depth=args.depth_loss_weight,
        weight_pose=args.pose_loss_weight,
    )

    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    global_step = 0
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        logger.info("\n--- Epoch {}/{} ---".format(epoch, args.epochs))
        avg_loss, avg_depth, avg_pose, global_step = train_one_epoch(
            model, dataloader, optimizer, scheduler, criterion, scaler,
            device, args, epoch, writer, global_step,
        )

        logger.info(
            "Epoch {} Summary: Loss={:.4f} Depth={:.4f} Pose={:.4f}".format(
                epoch, avg_loss, avg_depth, avg_pose)
        )
        if writer is not None:
            writer.add_scalar("epoch/loss", avg_loss, epoch)
            writer.add_scalar("epoch/depth_loss", avg_depth, epoch)
            writer.add_scalar("epoch/pose_loss", avg_pose, epoch)

        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, "lora_epoch_{}.pt".format(epoch))
            save_lora_weights(model, ckpt_path)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(args.output_dir, "lora_best.pt")
                save_lora_weights(model, best_path)
                logger.info("New best model saved (loss={:.4f})".format(best_loss))

    final_path = os.path.join(args.output_dir, "lora_final.pt")
    save_lora_weights(model, final_path)
    logger.info("Training complete. Final model saved to {}".format(final_path))

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
