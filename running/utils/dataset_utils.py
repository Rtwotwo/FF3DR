import os
import torch
from torch.utils.data import Dataset
import numpy as np 
import cv2
from PIL import Image
import OpenEXR
import Imath
from pathlib import Path
import logging
logger = logging.getLogger(__name__)


class WHUOMVSDataset(Dataset):
    """
    WHU-OMVS dataset loader for training, test and inference.
    Loads images, depths, masks, normals and camera parameters.
    For each sample, loads data from multiple cameras of the same frame.
    Args:
        dataset_dir: path to the dataset directory
        split: 'train' or 'test'
        num_views: number of views to load for each sample (default: 5)
        max_depth: maximum depth value to consider (for depth normalization)
    """
    def __init__(self, dataset_dir, 
                 split='train',
                 num_views=5,
                 max_depth=100.0):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.num_views = num_views
        self.max_depth = max_depth
        # read areas list
        index_file = self.dataset_dir / split / 'index.txt'
        with open(index_file, 'r') as f:
            self.areas = [line.strip() for line in f.readlines() if line.strip()]
        # load all samples containing images, depths, masks, normals and camera parameters
        self.samples = []
        for area in self.areas:
            area_dir = self.dataset_dir / split / area
            self._load_samples_from_area(area_dir)
        logger.info(f'[INFO] {len(self.samples)} samples loaded from {len(self.areas)} areas.')
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Return the sample at the given index.
        Args:
            idx: index of the sample to return
        
        Returns:
            sample: a dictionary containing images, depths, masks, normals and camera parameters
        """
        sample = self.samples[idx]
        image = self._load_image(sample['image_file'])
        cam_params = self._load_camera_params(sample['cam_file'])
        depth = self._load_depth(sample['depth_file'])
        normals = self._load_normal(sample['normal_file'])

        if sample['normal_file'] is not None:
            mask = self._load_mask(sample['mask_file'])
        else:
            mask = torch.ones(1, image.shape[1], image.shape[2])  

        # load multiview sample, default to load 5 views from the same frame
        if self.num_views > 1:
            src_images, src_cams, src_depths, src_masks, src_normals = self._load_source_views(sample, self.num_views - 1)
        else:
            src_images = torch.empty(0, 3, image.shape[1], image.shape[2])
            src_cams = torch.empty(0, 4, 4)
            src_depths = torch.empty(0, 1, depth.shape[1], depth.shape[2])
            src_masks = torch.empty(0, 1, mask.shape[1], mask.shape[2])
            src_normals = torch.empty(0, 3, image.shape[1], image.shape[2])

        return {
            'ref_image': image,
            'ref_cam': cam_params['extrinsic'],
            'ref_depth': depth,
            'ref_mask': mask,
            'ref_normal': normals,
            'ref_intrinsic': cam_params['intrinsic'],
            'src_images': src_images,
            'src_cams': src_cams,
            'src_depths': src_depths,
            'src_masks': src_masks,
            'src_normals': src_normals,
            'filename': f"{sample['area']}_{sample['sequence']}_{sample['frame_idx']:03d}"
        }
    
    def _load_samples_from_area(self, area_dir):
        """
        Load images, depths, masks, normals and camera parameters from a single area.
        Args:
            area_dir: path to the area directory containing samples
        """
        cams_dir = area_dir / 'cams'
        depths_dir = area_dir / 'depths'
        images_dir = area_dir / 'images'
        masks_dir = area_dir / 'masks'
        normals_dir = area_dir / 'normals'

        if not cams_dir.exists() or not depths_dir.exists() or not images_dir.exists() \
            or not masks_dir.exists() or not normals_dir.exists():
            logger.error(f'[ERROR] Missing directories in {area_dir}, skipping this area.')
            return

        # read all sequences in the area
        sequences = sorted([d for d in images_dir.iterdir() if d.is_dir()],
                            key=lambda x: int(x.name))
        for seq_dir in sequences:
            # get camera index to load data
            seq_id = seq_dir.name
            seq_cams_dir = cams_dir / seq_id
            seq_depths_dir = depths_dir / seq_id
            seq_images_dir = images_dir / seq_id
            seq_masks_dir = masks_dir / seq_id
            seq_normals_dir = normals_dir / seq_id
            # get all frames (cameras for this sequence/frame)
            cam_files = sorted(seq_cams_dir.glob('*.txt'))
            depth_files = sorted(seq_depths_dir.glob('*.exr'))
            image_files = sorted(seq_images_dir.glob('*.png'))
            mask_files = sorted(seq_masks_dir.glob('*.png'))
            normal_files = sorted(seq_normals_dir.glob('*.exr'))
            # check if the number of files is consistent
            min_files = min(len(cam_files), len(depth_files), len(image_files), 
                                len(mask_files), len(normal_files))
            # load data path for each camera/frame 
            for i in range(min_files):
                sample = {
                    'area': area_dir.name,
                    'sequence': seq_id,
                    'frame_idx': i,
                    'cam_file': cam_files[i],
                    'depth_file': depth_files[i],
                    'image_file': image_files[i],
                    'mask_file': mask_files[i],
                    'normal_file': normal_files[i]
                }
                self.samples.append(sample)
    
    def _load_camera_params(self, cam_path):
        """
        Load camera parameters from the given path.
        Args:
            cam_path: path to the camera parameter file
        """
        with open(cam_path, 'r') as f:
            lines = f.readlines()
        # read extrinsic matrix (4x4)
        extrinsic = []
        for i in range(1, 5):  
            extrinsic.append([float(x) for x in lines[i].strip().split()])
        extrinsic = torch.tensor(extrinsic, dtype=torch.float32)

        # read intrinsic matrix (3x3)
        intrinsic = []
        for i in range(6, 9):  
            intrinsic.append([float(x) for x in lines[i].strip().split()])
        intrinsic = torch.tensor(intrinsic, dtype=torch.float32)
        return {
            'extrinsic': extrinsic,
            'intrinsic': intrinsic
        }
    
    def _load_depth(self, depth_path):
        """
        Load depth map from the given path and normalize it.
        Args:
            depth_path: path to the depth map file
        """
        exr_file = OpenEXR.InputFile(str(depth_path))
        header = exr_file.header()
        dw = header['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        depth_str = exr_file.channel('Y', pt)  

        depth = np.frombuffer(depth_str, dtype=np.float32)
        depth = depth.reshape(size[1], size[0]) 

        depth = depth.copy()
        depth = torch.from_numpy(depth).unsqueeze(0)
        depth = torch.clamp(depth, 0, self.max_depth) / self.max_depth
        return depth
    
    def _load_image(self, image_path):
        """
        Load image from the given path and normalize it.
        Args:
            image_path: path to the image file
        """
        image = Image.open(image_path).convert('RGB')
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1) # HWC -> CHW
        return image
    
    def _load_mask(self, mask_path):
        """
        Load mask from the given path and normalize it.
        Args:
            mask_path: path to the mask file
        """
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0) # HW -> CHW
        return mask
    
    def _load_normal(self, normal_path):
        """
        Load normal map from the given path and normalize it.
        Args:
            normal_path: path to the normal map file
        """
        exr_file = OpenEXR.InputFile(str(normal_path))
        header = exr_file.header()
        dw = header['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        # Check available channels
        available_channels = header['channels'].keys()

        if 'X' in available_channels and 'Y' in available_channels and 'Z' in available_channels:
            normal_x_str = exr_file.channel('X', pt)  
            normal_y_str = exr_file.channel('Y', pt)  
            normal_z_str = exr_file.channel('Z', pt)
            normal_x = np.frombuffer(normal_x_str, dtype=np.float32).reshape(size[1], size[0])
            normal_y = np.frombuffer(normal_y_str, dtype=np.float32).reshape(size[1], size[0])
            normal_z = np.frombuffer(normal_z_str, dtype=np.float32).reshape(size[1], size[0])
            normal = np.stack([normal_x, normal_y, normal_z], axis=0)
        else:
            # Return default normal map (pointing upward) if channels not found
            logger.warning(f'[WARNING] No X, Y, Z channels in {normal_path}, using default normals.')
            normal = np.zeros((3, size[1], size[0]), dtype=np.float32)
            normal[2, :, :] = 1.0  # Z channel = 1.0 (pointing upward)
        normal = torch.from_numpy(normal).float()
        return normal
    
    def _load_source_views(self, ref_sample, num_src_views):
        """
        Load source views for the given reference sample from the same frame.
        Args:
            ref_sample: reference sample containing image, depth, mask, normal and camera parameters
            num_src_views: number of source views to load
        """
        area = ref_sample['area']
        sequence = ref_sample['sequence']
        ref_frame = ref_sample['frame_idx']
        # get source view indices (select other cameras from the same frame)
        area_dir = self.dataset_dir / self.split / area
        cams_dir = area_dir / 'cams' / sequence
        depths_dir = area_dir / 'depths' / sequence
        images_dir = area_dir / 'images' / sequence
        masks_dir = area_dir / 'masks' / sequence
        normals_dir = area_dir / 'normals' / sequence
        # get all cameras for this sequence/frame
        image_files = sorted(images_dir.glob('*.png'))
        total_cameras = len(image_files)

        available_cameras = [i for i in range(total_cameras) if i != ref_frame]
        if len(available_cameras) < num_src_views:
            selected_cameras = available_cameras + available_cameras[:num_src_views - len(available_cameras)]
        else:
            selected_cameras = np.random.choice(available_cameras, num_src_views, replace=False)

        src_images = []
        src_cams = []
        src_depths = []
        src_masks = []
        src_normals = []

        for cam_idx in selected_cameras:
            img_path = image_files[cam_idx]
            filename_stem = img_path.stem  
            src_img = self._load_image(img_path)
            src_images.append(src_img)
            # load camera parameters
            cam_path = cams_dir / f"{filename_stem}.txt"
            if cam_path.exists():
                cam_params = self._load_camera_params(cam_path)
                src_cams.append(cam_params['extrinsic'])
            # load depth map
            depth_path = depths_dir / f"{filename_stem}.exr"
            if depth_path.exists():
                src_depth = self._load_depth(depth_path)
                src_depths.append(src_depth)
            # load mask
            mask_path = masks_dir / f"{filename_stem}.png"
            if mask_path.exists():
                src_mask = self._load_mask(mask_path)
                src_masks.append(src_mask)
            # load normal map
            normal_path = normals_dir / f"{filename_stem}.exr"
            if normal_path.exists():
                src_normal = self._load_normal(normal_path)
                src_normals.append(src_normal)
        # convert lists to tensors
        if src_images:
            src_images = torch.stack(src_images)
        else:
            src_images = torch.empty(0, 3, 384, 768)

        if src_cams:
            src_cams = torch.stack(src_cams)
        else:
            src_cams = torch.empty(0, 4, 4)

        if src_depths:
            src_depths = torch.stack(src_depths)
        else:
            src_depths = torch.empty(0, 1, 384, 768)  

        if src_masks:
            src_masks = torch.stack(src_masks)
        else:
            src_masks = torch.empty(0, 1, 384, 768)  
        
        if src_normals:
            src_normals = torch.stack(src_normals)
        else:
            src_normals = torch.empty(0, 3, 384, 768)
        return src_images, src_cams, src_depths, src_masks, src_normals


def collate_fn(batch):
    """
    Collate function to handle variable number of source views.
    """
    # Find the maximum number of source views in the batch
    max_src_views = max(len(item['src_images']) for item in batch)

    for item in batch:
        # Pad source view data to the maximum number
        num_src = len(item['src_images'])
        if num_src < max_src_views:
            # Repeat the last source view
            if num_src > 0:
                pad_count = max_src_views - num_src
                item['src_images'] = torch.cat([item['src_images']] +
                                             [item['src_images'][-1:].repeat(pad_count, 1, 1, 1)])
                item['src_cams'] = torch.cat([item['src_cams']] +
                                            [item['src_cams'][-1:].repeat(pad_count, 1, 1)])
                item['src_depths'] = torch.cat([item['src_depths']] +
                                              [item['src_depths'][-1:].repeat(pad_count, 1, 1, 1)])
                item['src_masks'] = torch.cat([item['src_masks']] +
                                             [item['src_masks'][-1:].repeat(pad_count, 1, 1, 1)])
                item['src_normals'] = torch.cat([item['src_normals']] +
                                               [item['src_normals'][-1:].repeat(pad_count, 1, 1, 1)])
            else:
                # If no source views, create empty tensors
                item['src_images'] = torch.zeros(max_src_views, 3, 384, 768)
                item['src_cams'] = torch.zeros(max_src_views, 4, 4)
                item['src_depths'] = torch.zeros(max_src_views, 1, 384, 768)
                item['src_masks'] = torch.zeros(max_src_views, 1, 384, 768)
                item['src_normals'] = torch.zeros(max_src_views, 3, 384, 768)
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == '__main__':
    dataset = WHUOMVSDataset(
        dataset_dir="/data2/dataset/Redal/work_feedforward_3drepo/dataset/WHU-OMVS",
        split='train',
        num_views=5)
    print(f"Dataset size: {len(dataset)}")
    # Test loading one sample
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