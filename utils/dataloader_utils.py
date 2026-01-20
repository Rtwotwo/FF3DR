"""
Author: Redal
Date: 2025-12-23
Todo: 读取WHU-OMVS数据集的相机参数,深度图以及掩码等
      相机的extrisics和intrisics参数主要在每个场景的info文件夹
Homepage: https://github.com/Rtwotwo/FF3DR.git
"""
import os
import re
import sys
import numpy as np
from typing import Dict, List,Tuple, Optional


class Camera:
    """注册相机内部参数包括id,尺寸,像素大小,焦距,主点,畸变参数等"""
    def __init__(self,
                camera_id: Optional[int] = None,
                size: Optional[Tuple[int, int]] = None,               
                pixelsize: Optional[float] = None,
                focallength: Optional[Tuple[float, float]] = None,    
                x0y0: Optional[Tuple[float, float]] = None,           
                distortion: Optional[List[float]] = None              
                ) -> None:
        self.camera_id = camera_id
        self.size = size # [width, height]
        self.pixelsize = pixelsize        
        self.focallength = focallength # [fx, fy]
        self.x0y0 = x0y0 # [cx, cy]
        self.distortion = distortion # [k1, k2, k3, p1, p2] (length=5)
    def __lt__(self,):
        return [self.camera_id, self.size, self.pixelsize, 
                self.focallength, self.x0y0, self.distortion]
    

class Photo:
    """注册图像的位姿参数包括id,相机id,旋转矩阵,投影中心,深度范围,名称等"""
    def __init__(self,
                image_id: Optional[int] = None,
                camera_id: Optional[int] = None,
                rotation_matrix: Optional[np.ndarray] = None,  
                project_center: Optional[np.ndarray] = None,    
                depth: Optional[Tuple[float, float]] = None,    
                name: Optional[str] = None,                      
                camera_coordinate_type: str = 'XrightYup',      
                rotation_type: str = 'Rwc',                      
                translation_type: str = 'twc'                    
                ) -> None:
        self.image_id = image_id
        self.camera_id = camera_id
        self.name = name
        self.rotation_matrix = rotation_matrix  # Rwc [3,3]
        self.project_center = project_center    # twc [x,y,z]
        self.depth = depth                      # [mindepth, maxdepth]
        self.camera_coordinate_type = camera_coordinate_type
        self.rotation_type = rotation_type
        self.translation_type = translation_type
    def __lt__(self,):
        return [self.image_id, self.camera_id,
                self.rotation_matrix, self.project_center,
                self.depth, self.name]


def read_cameras_text(path: str):
    """读取相机参数文本文件,主要包括的相机内参如下CAMERA_ID, WIDTH, 
    HEIGHT, PIXELSIZE, PARAMS[fx,fy,cx,cy], DISTORTION[K1, K2, K3, P1, P2]"""
    cams = {}
    with open(path, 'r') as fid:
        while True:
            line = fid.readline()
            if not line: break
            line = line.strip()
            if len(line) > 0 and line[0] != '#':
                elems = line.split()
                camera_id = int(elems[0])
                width = int(elems[1])
                height = int(elems[2])
                pixelsize = float(elems[3])
                params = np.array(tuple(map(float, elems[4:8])))
                distortion = np.array(tuple(map(float, elems[8:])))
                cams[camera_id] = Camera(camera_id=camera_id,
                                        size=(width, height),
                                        pixelsize=pixelsize,
                                        focallength=(params[0], params[1]),
                                        x0y0=(params[2], params[3]),
                                        distortion=distortion)
    return cams


def read_images_text(path:str):
    """读取图像参数文本文件,主要包括的图像位姿参数如下IMAGE_ID, CAMERA_ID, 
    Rwc[9], twc[3], MINDEPTH, MAXDEPTH, NAME"""
    images = {}
    with open(path, 'r') as fid:
        while True:
            line = fid.readline()
            if not line: break
            line = line.strip()
            if len(line) > 0 and line[0] != '#':
                elems = line.split()
                image_id = int(elems[0])
                Camera_id = int(elems[1])
                R_matrix = np.array(tuple(map(float, elems[2:11]))).reshape(3, 3)
                t_matrix = np.array(tuple(map(float, elems[11:14])))
                depth_range = np.array(tuple(map(float, elems[14:16])))
                image_name = elems[16]
                images[image_id] = Photo(image_id=image_id,
                                         camera_id=Camera_id,
                                         rotation_matrix=R_matrix,
                                         project_center=t_matrix,
                                         depth=(depth_range[0], depth_range[1]),
                                         name=image_name)
    return images


def read_images_path_text(path:str, replace_dir:bool=True):
    """读取图像路径文本文件,主要包括IMAGE_ID和IMAGE_PATH
    replace_dir: 是否替换路径中的某些目录以适应当前环境"""
    paths_list = {}
    names_list = {}
    cluster_list = open(path).read().split()
    total_num = int(cluster_list[0])
    for i in range(total_num):
        index = int (cluster_list[i * 3 + 1])  
        name = cluster_list[i * 3 + 2]
        p = cluster_list[i * 3 + 3]
        if replace_dir:
            # 修改图像存储路径
            p = p.split('/')[-4:]
            path_dir = path.split('/')[:-3]
            p = os.path.join('/'.join(path_dir), '/'.join(p))
        paths_list[index] = p
        names_list[index] = name
    return paths_list, names_list


def read_view_pair_text(pair_path:str, view_num:int):
    """读取视图对文本文件,主要包括每个样本的视图索引"""
    metas = []
    # read the pair file
    with open(pair_path) as f:
        num_viewpoint = int(f.readline())
        # viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = [int(f.readline().rstrip())]
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            # filter by no src view and fill to nviews
            if len(src_views) > 0:
                if len(src_views) < view_num:
                    print("{}< num_views:{}".format(len(src_views), view_num))
                    src_views += [src_views[0]] * (view_num - len(src_views))
                metas.append(ref_view + src_views)
    return metas


def write_red_cam(file, cam, ref_path):
    """将相机的外参extrinsic和内参intrinsic以
    一种自定义文本格式称为RED格式写入文件"""
    f = open(file, "w")
    f.write('extrinsic: XrightYdown, [Rcw|tcw]\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')
    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')
    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + 
            ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')
    f.write('\n')
    f.write(str(ref_path) + '\n')
    f.close()


def read_pfm(filename):
    """读取PFM格式的Portable FloatMap格式的深度图或视差图"""
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None
    # 获得深度图的头文件信息
    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else: raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else: raise Exception('Malformed PFM header.')
    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else: endian = '>'  # big-endian
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):
    """将numpy浮点数组保存为PFM格式,与read_pfm配对使用"""
    file = open(filename, "wb")
    color = None
    image = np.flipud(image)
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')
    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')
    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))
    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale
    file.write(('%f\n' % scale).encode('utf-8'))
    image.tofile(file)
    file.close()