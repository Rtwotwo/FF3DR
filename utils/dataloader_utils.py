"""
Author: Redal
Date: 2025-12-23
Todo: 读取WHU-OMVS数据集的相机参数,深度图以及掩码等
      相机的extrisics和intrisics参数主要在每个场景的info文件夹
Homepage: https://github.com/Rtwotwo/FF3DR.git
"""
import os
import numpy as np


def read_camera_info(camera_info_path:str):
    """读取WHU_OMVS数据集中每个场景的相机参数,包括
    CAMERA_ID, WIDTH, HEIGHT, PIXELSIZE, PARAMS[fx,fy,cx,cy], DISTORTION[K1, K2, K3, P1, P2]
    存储该文件的相对路径位于: WHU-OMVS/train/area1/info/camera_info.txt
    camera_info_path输入相机的相关路径;Return返回存储五组相机视角的参数信息返回五组相机"""
    with open(camera_info_path, 'r') as text_file:
        camera_info_lines = text_file.readlines()
    # 分割参数形成内外参数信息
    CAMERAS_INFO = {}
    for line in camera_info_lines[4:]:
        camera_info = {}
        line_data = list(map(float, line.split()))
        camera_info['camera_id'] = int(line_data[0])
        camera_info['image_size'] = line_data[1:3]
        camera_info['pixel_size'] = line_data[3]
        camera_info['f_xy'] = line_data[4:6]
        camera_info['c_xy'] = line_data[6:8]
        camera_info['k_123'] = line_data[8:11]
        camera_info['p_12'] = line_data[11:13]
        CAMERAS_INFO[f'{int(line_data[0])}'] = camera_info
    print(f'[INFO] the camera_info path: {camera_info_path} has been read out! All cameras number: {len(CAMERAS_INFO)}')
    return CAMERAS_INFO


def read_image_info(image_info_path:str):
    """读取图片数据的位姿参数,主要包括:
    IMAGE_ID, CAMERA_ID, Rwc[9], twc[3], MINDEPTH, MAXDEPTH, NAME
    存储该文件的相对路径位于: WHU-OMVS/train/area1/info/image_info.txt
    image_info_path存储图像参数的路径;Return包含图像位姿参数的字典"""
    with open(image_info_path, 'r') as image_file:
        image_info_data = image_file.readlines()
    # 数据转换成字典形式
    IMAGES_INFO = {}
    for line in image_info_data[4:]:
        parts = line.split()
        line_data = list(map(float, parts[:16]))
        img_name = str(parts[16])
        IMAGES_INFO[img_name] = {
            'image_id': int(line_data[0]),
            'camera_id': int(line_data[1]),
            'Rwc': line_data[2:11],              
            'twc': line_data[11:14],             
            'min_depth': line_data[14],
            'max_depth': line_data[15],
            'name': img_name}
    print(f'[INFO] the image info path: {image_info_path} has been read out! All image number: {len(IMAGES_INFO)}')
    return IMAGES_INFO


def read_exr_file(exr_file_path:str):
    """exr_file读取深度图信息,主要包括每个视角的深度图
    将深度图的exr文件转成后续用于计算的格式,便于操作"""
    
