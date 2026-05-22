from torch.utils.data import Dataset
import numpy as np
import os
from .preprocess import *
from .data_io import *
from imageio import imread, imsave, imwrite

"""
WHU-OMVS test split dataset for Ada-MVS inference.
Handles the test directory layout:
  test/{area}/info/{camera_info,image_info,image_path,viewpair}.txt
  test/{area}/images/{cam_id}/{stem}.png
  test/{area}/depths/{cam_id}/{stem}.exr
  test/{area}/masks/{cam_id}/{stem}.png
  test/{area}/cams/{cam_id}/{stem}.txt

Key difference from predict_oblique:
  - image_path.txt contains Windows absolute paths (H:/project/...),
    so we remap them to the actual local paths.
  - data_folder points to test/{area}/info instead of predict/source.
"""


class MVSDataset(Dataset):
    def __init__(self, data_folder, view_num, args):
        super(MVSDataset, self).__init__()
        self.data_folder = data_folder
        self.viewpair_path = os.path.join(data_folder, 'viewpair.txt')
        self.image_params_path = os.path.join(data_folder, 'image_info.txt')
        self.cam_params_path = os.path.join(data_folder, 'camera_info.txt')
        self.image_path_path = os.path.join(data_folder, 'image_path.txt')

        self.args = args
        self.view_num = view_num
        self.min_interval = args.min_interval
        self.interval_scale = args.interval_scale
        self.num_depth = args.numdepth
        self.counter = 0

        self.area_root = os.path.dirname(data_folder)

        self.cam_params_dict = read_cameras_text(self.cam_params_path)
        self.image_params_dict = read_images_text(self.image_params_path)
        self.image_paths, self.image_names = read_images_path_text(self.image_path_path)
        self._remap_image_paths()
        self.sample_list = read_view_pair_text(self.viewpair_path, self.view_num)
        self.sample_list = self._limit_samples_per_camera(
            self.sample_list,
            getattr(args, "test_max_samples_per_camera", 20),
        )
        self.sample_num = len(self.sample_list)

    def _limit_samples_per_camera(self, sample_list, max_samples_per_camera):
        if max_samples_per_camera is None or int(max_samples_per_camera) <= 0:
            return sample_list

        max_samples_per_camera = int(max_samples_per_camera)
        camera_counts = {}
        limited_samples = []

        for sample in sample_list:
            ref_view_idx = sample[0]
            image_params = self.image_params_dict.get(ref_view_idx, None)
            if image_params is None:
                continue

            camera_id = str(image_params.camera_id)
            current_count = camera_counts.get(camera_id, 0)
            if current_count >= max_samples_per_camera:
                continue

            limited_samples.append(sample)
            camera_counts[camera_id] = current_count + 1

        return limited_samples

    def _remap_image_paths(self):
        remapped = {}
        for idx, path in self.image_paths.items():
            name = self.image_names.get(idx, "")
            parts = name.split("/")
            if len(parts) >= 2:
                cam_id = parts[0]
                stem = parts[1]
                local_path = os.path.join(self.area_root, "images", cam_id, stem + ".png")
                remapped[idx] = local_path
            else:
                basename = os.path.basename(path)
                cam_id = basename.split("_")[0] if "_" in basename else "1"
                local_path = os.path.join(self.area_root, "images", cam_id, basename + ".png")
                remapped[idx] = local_path
        self.image_paths = remapped

    def __len__(self):
        return len(self.sample_list)

    def read_img(self, filename):
        img = Image.open(filename)
        return img

    def read_depth(self, filename):
        depth_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        mask_path = filename.replace("depths", "masks")
        mask_path = mask_path.replace(".exr", ".png")
        mask_image = np.array(cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)) / 255.
        mask_image = mask_image < 0.5
        depth_image[mask_image] = 0
        return np.array(depth_image)

    def center_image(self, img, mode='mean'):
        if mode == 'standard':
            np_img = np.array(img, dtype=np.float32) / 255.
        elif mode == 'mean':
            img_array = np.array(img)
            img = img_array.astype(np.float32)
            var = np.var(img, axis=(0, 1), keepdims=True)
            mean = np.mean(img, axis=(0, 1), keepdims=True)
            np_img = (img - mean) / (np.sqrt(var) + 0.00000001)
        else:
            raise Exception("{}? Not implemented yet!".format(mode))
        return np_img

    def create_cams(self, image_params, cam_params_dict, num_depth=384, min_interval=0.1):
        cam = np.zeros((2, 4, 4), dtype=np.float32)
        extrinsics = np.zeros((4, 4), dtype=np.float32)

        O_xrightyup = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        R = np.matmul(image_params.rotation_matrix, O_xrightyup)
        t = image_params.project_center
        extrinsics[0:3, 0:3] = R
        extrinsics[0:3, 3] = t
        extrinsics[3, 3] = 1.0
        extrinsics = np.linalg.inv(extrinsics)
        cam[0, :, :] = extrinsics

        cam_params = cam_params_dict[image_params.camera_id]
        fx = cam_params.focallength[0]
        fy = cam_params.focallength[1]
        x0 = cam_params.x0y0[0]
        y0 = cam_params.x0y0[1]

        cam[1][0][0] = fx
        cam[1][1][1] = fy
        cam[1][0][2] = x0
        cam[1][1][2] = y0
        cam[1][2][2] = 1

        cam[1][3][0] = image_params.depth[0]
        cam[1][3][1] = (image_params.depth[1] - image_params.depth[0]) / num_depth
        cam[1][3][3] = image_params.depth[1]
        cam[1][3][2] = num_depth

        return cam

    def __getitem__(self, idx):
        data = self.sample_list[idx]

        outimage = None
        outcam = None

        centered_images = []
        proj_matrices = []

        for view in range(self.view_num):
            image_idx = data[view]
            image = self.read_img(self.image_paths[image_idx])
            image = np.array(image)

            depth_interval = self.min_interval * self.interval_scale
            image_params = self.image_params_dict[image_idx]
            cam = self.create_cams(image_params, self.cam_params_dict, self.num_depth, depth_interval)

            scaled_image, scaled_cam = scale_input(image, cam, scale=self.args.resize_scale)
            croped_image, croped_cam = crop_input(scaled_image, scaled_cam, max_h=self.args.max_h,
                                                  max_w=self.args.max_w, resize_scale=self.args.resize_scale)

            if view == 0:
                ref_img_path = self.image_paths[image_idx]
                outimage = croped_image
                outcam = croped_cam
                depth_min = croped_cam[1][3][0]
                depth_max = croped_cam[1][3][3]
                image_name = image_params.name
                h, w = croped_image.shape[0:2]

            scaled_cam = scale_camera(croped_cam, scale=self.args.sample_scale)
            extrinsics = scaled_cam[0, :, :]
            intrinsics = scaled_cam[1, 0:3, 0:3]
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])

            proj_matrices.append(proj_mat)
            centered_images.append(self.center_image(croped_image))

        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        depth_values = np.array([depth_min, depth_max], dtype=np.float32)

        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, :2, :] = proj_matrices[:, :2, :] / 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, :2, :] = proj_matrices[:, :2, :] / 4

        proj_matrices_ms = {
            "stage1": stage3_pjmats,
            "stage2": stage2_pjmats,
            "stage3": proj_matrices
        }

        name = os.path.splitext(os.path.basename(image_name))[0]
        vid = os.path.dirname(image_name).split("/")[-1]

        return {"imgs": centered_images,
                "proj_matrices": proj_matrices_ms,
                "depth_values": depth_values,
                "outimage": outimage,
                "outcam": outcam,
                "ref_image_path": ref_img_path,
                "out_name": name,
                "out_view": vid}
