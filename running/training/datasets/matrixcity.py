from __future__ import annotations

import json
import math
import os
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)


class MatrixCityDataset(Dataset):
    """
    MatrixCity dataset loader for DA3 LoRA fine-tuning.

    Loads multi-view aerial images with camera poses and depth maps.
    Supports both big_city and small_city variants.

    Dataset structure:
        MatrixCity/
        ├── big_city/aerial/train/{block}/
        │   ├── {frame:04d}.png          # RGB images
        │   └── transforms.json          # Camera poses (rot_mat 4x4 per frame)
        ├── big_city_depth/aerial/train/{block}/
        │   └── {frame:04d}.exr          # Depth maps
        ├── small_city/aerial/train/{block}/
        └── small_city_depth/aerial/train/{block}/
    """

    def __init__(
        self,
        dataset_dir,
        city_size="big_city",
        split="train",
        num_views=2,
        max_depth=500.0,
        image_size=504,
        stride=10,
        max_samples_per_block=-1,
        depth_supervision=True,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.city_size = city_size
        self.split = split
        self.num_views = num_views
        self.max_depth = max_depth
        self.image_size = image_size
        self.stride = stride
        self.depth_supervision = depth_supervision

        self.image_dir = self.dataset_dir / city_size / "aerial" / split
        self.depth_dir = self.dataset_dir / "{}_depth".format(city_size) / "aerial" / split

        self.samples = []
        self._build_sample_list(max_samples_per_block)

        logger.info(
            "[MatrixCity] Loaded {} samples from {}/{}, num_views={}, stride={}".format(
                len(self.samples), city_size, split, num_views, stride)
        )

    def _build_sample_list(self, max_samples_per_block):
        if not self.image_dir.exists():
            raise FileNotFoundError("Image directory not found: {}".format(self.image_dir))

        blocks = sorted([d for d in self.image_dir.iterdir() if d.is_dir()])
        for block_dir in blocks:
            block_name = block_dir.name
            transforms_path = block_dir / "transforms.json"
            if not transforms_path.exists():
                logger.warning("No transforms.json in {}, skipping".format(block_dir))
                continue

            with open(transforms_path, "r") as f:
                transforms = json.load(f)

            camera_angle_x = transforms.get("camera_angle_x", 0.785)
            frames = transforms.get("frames", [])

            if len(frames) < self.num_views:
                logger.warning("Block {} has {} frames, need {}, skipping".format(
                    block_name, len(frames), self.num_views))
                continue

            sampled_indices = list(range(0, len(frames), self.stride))
            if max_samples_per_block > 0 and len(sampled_indices) > max_samples_per_block:
                sampled_indices = random.sample(sampled_indices, max_samples_per_block)

            for idx in sampled_indices:
                view_indices = self._select_views(idx, len(frames))

                sample = {
                    "block": block_name,
                    "camera_angle_x": camera_angle_x,
                    "view_indices": view_indices,
                    "view_frames": [frames[i] for i in view_indices],
                    "num_total_frames": len(frames),
                }
                self.samples.append(sample)

    def _select_views(self, ref_idx, total_frames):
        if self.num_views == 1:
            return [ref_idx]

        candidates = []
        offsets = [1, 2, 3, 5, 8, 13, 21, 34]
        for off in offsets:
            for sign in [1, -1]:
                candidate = ref_idx + sign * off
                if 0 <= candidate < total_frames and candidate != ref_idx:
                    candidates.append(candidate)

        random.shuffle(candidates)
        selected = [ref_idx] + candidates[:self.num_views - 1]

        if len(selected) < self.num_views:
            remaining = [i for i in range(total_frames) if i not in selected]
            random.shuffle(remaining)
            selected += remaining[:self.num_views - len(selected)]

        return selected[:self.num_views]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        block_name = sample["block"]
        camera_angle_x = sample["camera_angle_x"]
        view_frames = sample["view_frames"]

        images = []
        extrinsics = []
        intrinsics = []
        depths = []

        block_image_dir = self.image_dir / block_name
        block_depth_dir = self.depth_dir / block_name

        for frame_info in view_frames:
            frame_idx = frame_info["frame_index"]
            rot_mat = np.array(frame_info["rot_mat"], dtype=np.float32)

            img_path = block_image_dir / "{:04d}.png".format(frame_idx)
            image = self._load_image(img_path)
            images.append(image)

            ext = torch.from_numpy(rot_mat)
            extrinsics.append(ext)

            ixt = self._compute_intrinsics(camera_angle_x, image.shape[-1], image.shape[-2])
            intrinsics.append(ixt)

            if self.depth_supervision:
                depth_path = block_depth_dir / "{:04d}.exr".format(frame_idx)
                depth = self._load_depth(depth_path)
                depths.append(depth)

        images = torch.stack(images)
        extrinsics = torch.stack(extrinsics)
        intrinsics = torch.stack(intrinsics)

        result = {
            "images": images,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "block": block_name,
        }

        if self.depth_supervision and len(depths) > 0:
            result["depths"] = torch.stack(depths)

        return result

    def _load_image(self, path):
        img = Image.open(str(path)).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img

    def _load_depth(self, path):
        if not path.exists():
            return torch.zeros(1, self.image_size, self.image_size)

        try:
            import OpenEXR
            import Imath

            exr_file = OpenEXR.InputFile(str(path))
            header = exr_file.header()
            dw = header["dataWindow"]
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            depth_str = exr_file.channel("Y", pt)
            depth = np.frombuffer(depth_str, dtype=np.float32).reshape(size[1], size[0])
            depth = depth.copy()

            depth = torch.from_numpy(depth).unsqueeze(0)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            depth = torch.clamp(depth, 0, self.max_depth) / self.max_depth
            return depth
        except Exception as e:
            logger.warning("Failed to load depth {}: {}".format(path, e))
            return torch.zeros(1, self.image_size, self.image_size)

    def _compute_intrinsics(self, camera_angle_x, w, h):
        fx = w / (2.0 * math.tan(camera_angle_x / 2.0))
        fy = fx
        cx = w / 2.0
        cy = h / 2.0

        intrinsic = torch.tensor(
            [[fx, 0, cx],
             [0, fy, cy],
             [0, 0, 1]],
            dtype=torch.float32,
        )
        return intrinsic


def collate_fn(batch):
    result = {
        "images": torch.stack([item["images"] for item in batch]),
        "extrinsics": torch.stack([item["extrinsics"] for item in batch]),
        "intrinsics": torch.stack([item["intrinsics"] for item in batch]),
        "block": [item["block"] for item in batch],
    }

    if "depths" in batch[0]:
        result["depths"] = torch.stack([item["depths"] for item in batch])

    return result
