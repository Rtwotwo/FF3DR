import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
import OpenEXR
import Imath
from pathlib import Path


class WHU_OMVS_Dataset(Dataset):
    """
    WHU-OMVS数据集加载器

    数据集结构：
    - train/
        - area1/, area4/, area5/...
            - images/1/, images/2/, ... (每个序列52帧PNG图像)
            - cams/1/, cams/2/, ... (每个序列52个相机参数txt文件)
            - depths/1/, depths/2/, ... (每个序列52个EXR深度图)
            - masks/ (可选)
            - normals/ (可选)
            - info/
                - camera_info.txt
                - viewpair.txt
    """
    def __init__(self, root_dir, split='train', num_views=5, max_depth=100.0):
        """
        初始化数据集

        Args:
            root_dir: 数据集根目录
            split: 'train', 'test', 'predict'
            num_views: 每个样本使用的视角数量
            max_depth: 最大深度值，用于归一化
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_views = num_views
        self.max_depth = max_depth

        # 读取区域列表
        index_file = self.root_dir / split / 'index.txt'
        with open(index_file, 'r') as f:
            self.areas = [line.strip() for line in f.readlines() if line.strip()]

        # 收集所有样本
        self.samples = []
        for area in self.areas:
            area_dir = self.root_dir / split / area
            self._collect_samples_from_area(area_dir)

        print(f"Found {len(self.samples)} samples in {len(self.areas)} areas")

    def _collect_samples_from_area(self, area_dir):
        """从一个区域收集样本"""
        images_dir = area_dir / 'images'
        cams_dir = area_dir / 'cams'
        depths_dir = area_dir / 'depths'
        masks_dir = area_dir / 'masks'

        if not images_dir.exists() or not cams_dir.exists() or not depths_dir.exists():
            print(f"Warning: Missing required directories in {area_dir}")
            return

        # 获取所有序列
        sequences = sorted([d for d in images_dir.iterdir() if d.is_dir()],
                          key=lambda x: int(x.name))

        for seq_dir in sequences:
            seq_id = seq_dir.name

            # 检查对应的目录是否存在
            seq_cams_dir = cams_dir / seq_id
            seq_depths_dir = depths_dir / seq_id
            seq_masks_dir = masks_dir / seq_id if masks_dir.exists() else None

            if not seq_cams_dir.exists() or not seq_depths_dir.exists():
                continue

            # 获取该序列的所有帧
            image_files = sorted(seq_dir.glob('*.png'))
            cam_files = sorted(seq_cams_dir.glob('*.txt'))
            depth_files = sorted(seq_depths_dir.glob('*.exr'))
            mask_files = sorted(seq_masks_dir.glob('*.png')) if seq_masks_dir else []

            # 确保文件数量匹配
            min_files = min(len(image_files), len(cam_files), len(depth_files))
            if seq_masks_dir:
                min_files = min(min_files, len(mask_files))

            for i in range(min_files):
                sample = {
                    'area': area_dir.name,
                    'sequence': seq_id,
                    'frame_idx': i,
                    'image_path': image_files[i],
                    'cam_path': cam_files[i],
                    'depth_path': depth_files[i],
                    'mask_path': mask_files[i] if mask_files else None
                }
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载图像
        image = self._load_image(sample['image_path'])

        # 加载相机参数
        cam_params = self._load_camera_params(sample['cam_path'])

        # 加载深度图
        depth = self._load_depth(sample['depth_path'])

        # 加载掩码（如果存在）
        if sample['mask_path'] is not None:
            mask = self._load_mask(sample['mask_path'])
        else:
            mask = torch.ones(1, image.shape[1], image.shape[2])  # 默认全1掩码

        # 加载多视角数据（如果需要）
        if self.num_views > 1:
            src_images, src_cams, src_depths, src_masks = self._load_source_views(sample, self.num_views - 1)
        else:
            src_images = torch.empty(0, 3, image.shape[1], image.shape[2])
            src_cams = torch.empty(0, 4, 4)
            src_depths = torch.empty(0, 1, depth.shape[1], depth.shape[2])
            src_masks = torch.empty(0, 1, mask.shape[1], mask.shape[2])

        return {
            'ref_image': image,
            'ref_cam': cam_params['extrinsic'],
            'ref_depth': depth,
            'ref_mask': mask,
            'ref_intrinsic': cam_params['intrinsic'],
            'src_images': src_images,
            'src_cams': src_cams,
            'src_depths': src_depths,
            'src_masks': src_masks,
            'filename': f"{sample['area']}_{sample['sequence']}_{sample['frame_idx']:03d}"
        }

    def _load_image(self, image_path):
        """加载图像"""
        image = Image.open(image_path).convert('RGB')
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        return image

    def _load_camera_params(self, cam_path):
        """加载相机参数"""
        with open(cam_path, 'r') as f:
            lines = f.readlines()

        # 解析外参矩阵 (4x4)
        extrinsic = []
        for i in range(1, 5):  # 跳过第一行标题
            extrinsic.append([float(x) for x in lines[i].strip().split()])
        extrinsic = torch.tensor(extrinsic, dtype=torch.float32)

        # 解析内参矩阵 (3x3)
        intrinsic = []
        for i in range(6, 9):  # 第6-8行
            intrinsic.append([float(x) for x in lines[i].strip().split()])
        intrinsic = torch.tensor(intrinsic, dtype=torch.float32)

        return {
            'extrinsic': extrinsic,
            'intrinsic': intrinsic
        }

    def _load_mask(self, mask_path):
        """加载掩码"""
        mask = Image.open(mask_path).convert('L')  # 转换为灰度图
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)  # 添加通道维度
        return mask

    def _load_depth(self, depth_path):
        """加载EXR深度图"""
        # 使用OpenEXR读取EXR文件
        exr_file = OpenEXR.InputFile(str(depth_path))
        header = exr_file.header()
        dw = header['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        # 读取深度通道
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        depth_str = exr_file.channel('Y', pt)  # WHU-OMVS使用Y通道存储深度

        # 转换为numpy数组
        depth = np.frombuffer(depth_str, dtype=np.float32)
        depth = depth.reshape(size[1], size[0])  # 注意：EXR是行优先

        # 确保数组可写
        depth = depth.copy()
        # 转换为torch tensor并归一化
        depth = torch.from_numpy(depth).unsqueeze(0)  # 添加通道维度
        # 归一化深度值
        depth = torch.clamp(depth, 0, self.max_depth) / self.max_depth
        return depth

    def _load_source_views(self, ref_sample, num_src_views):
        """加载源视角数据"""
        area = ref_sample['area']
        sequence = ref_sample['sequence']
        ref_frame = ref_sample['frame_idx']

        area_dir = self.root_dir / self.split / area
        images_dir = area_dir / 'images' / sequence
        cams_dir = area_dir / 'cams' / sequence
        depths_dir = area_dir / 'depths' / sequence
        masks_dir = area_dir / 'masks' / sequence

        # 获取该序列的所有图像文件
        image_files = sorted(images_dir.glob('*.png'))
        total_frames = len(image_files)

        # 选择源视角（避免选择参考帧）
        available_frames = [i for i in range(total_frames) if i != ref_frame]
        if len(available_frames) < num_src_views:
            # 如果不够，用重复的帧填充
            selected_frames = available_frames + available_frames[:num_src_views - len(available_frames)]
        else:
            # 随机选择
            selected_frames = np.random.choice(available_frames, num_src_views, replace=False)

        src_images = []
        src_cams = []
        src_depths = []
        src_masks = []

        for frame_idx in selected_frames:
            # 使用实际的文件名
            img_path = image_files[frame_idx]
            filename_stem = img_path.stem  # 不含扩展名的文件名，如 '001_001'

            # 加载源图像
            src_img = self._load_image(img_path)
            src_images.append(src_img)

            # 加载源相机参数
            cam_path = cams_dir / f"{filename_stem}.txt"
            if cam_path.exists():
                cam_params = self._load_camera_params(cam_path)
                src_cams.append(cam_params['extrinsic'])

            # 加载源深度
            depth_path = depths_dir / f"{filename_stem}.exr"
            if depth_path.exists():
                src_depth = self._load_depth(depth_path)
                src_depths.append(src_depth)

            # 加载源掩码
            mask_path = masks_dir / f"{filename_stem}.png"
            if mask_path.exists():
                src_mask = self._load_mask(mask_path)
                src_masks.append(src_mask)

        # 转换为tensor
        if src_images:
            src_images = torch.stack(src_images)
        else:
            src_images = torch.empty(0, 3, 384, 768)  # 默认尺寸

        if src_cams:
            src_cams = torch.stack(src_cams)
        else:
            src_cams = torch.empty(0, 4, 4)

        if src_depths:
            src_depths = torch.stack(src_depths)
        else:
            src_depths = torch.empty(0, 1, 384, 768)  # 默认尺寸

        if src_masks:
            src_masks = torch.stack(src_masks)
        else:
            src_masks = torch.empty(0, 1, 384, 768)  # 默认尺寸

        return src_images, src_cams, src_depths, src_masks


def collate_fn(batch):
    """
    自定义collate函数处理可变长度的数据
    """
    # 找到批次中的最大源视角数量
    max_src_views = max(len(item['src_images']) for item in batch)

    for item in batch:
        # 填充源视角数据到最大数量
        num_src = len(item['src_images'])
        if num_src < max_src_views:
            # 重复最后一个源视角
            if num_src > 0:
                pad_count = max_src_views - num_src
                item['src_images'] = torch.cat([item['src_images']] +
                                             [item['src_images'][-1:].repeat(pad_count, 1, 1, 1)])
                item['src_cams'] = torch.cat([item['src_cams']] +
                                            [item['src_cams'][-1:].repeat(pad_count, 1, 1)])
                item['src_depths'] = torch.cat([item['src_depths']] +
                                              [item['src_depths'][-1:].repeat(pad_count, 1, 1, 1)])
            else:
                # 如果没有源视角，创建空的tensor
                item['src_images'] = torch.zeros(max_src_views, 3, 384, 768)
                item['src_cams'] = torch.zeros(max_src_views, 4, 4)
                item['src_depths'] = torch.zeros(max_src_views, 1, 384, 768)

    # 使用默认的collate
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == "__main__":
    # 测试数据集
    dataset = WHU_OMVS_Dataset(
        root_dir="/data2/dataset/Redal/work_feedforward_3drepo/dataset/WHU-OMVS",
        split='train',
        num_views=5
    )

    print(f"Dataset size: {len(dataset)}")

    # 测试加载一个样本
    sample = dataset[0]
    print("Sample keys:", list(sample.keys()))
    print(f"Reference image shape: {sample['ref_image'].shape}")
    print(f"Reference camera shape: {sample['ref_cam'].shape}")
    print(f"Reference depth shape: {sample['ref_depth'].shape}")
    print(f"Reference mask shape: {sample['ref_mask'].shape}")
    print(f"Reference intrinsic shape: {sample['ref_intrinsic'].shape}")
    print(f"Source images shape: {sample['src_images'].shape}")
    print(f"Source cameras shape: {sample['src_cams'].shape}")
    print(f"Source depths shape: {sample['src_depths'].shape}")
    print(f"Source masks shape: {sample['src_masks'].shape}")
    print(f"Filename: {sample['filename']}")