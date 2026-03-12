import argparse
import gc
import glob
import json
import os
import shutil
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from utils.config_utils import load_config

# load models da3, mapanything, pi3, vggt
from safetensors.torch import load_file
from models.depthanything3.api import DepthAnything3
from models.mapanything.models.mapanything.model import MapAnything
from models.pi3.models.pi3 import Pi3
from models.vggt.models.vggt import VGGT


class FF3DR:
    """load all images from simgle area and make sure that putting 
    frame of images from 5 cameras and to load other data
    Args:
        area_path(str): images path from single area
        chunk_size(int): put chunk_size frames to inference 
    """
    def __init__(self, args):
        # Passing hyper-params for FF3DR 
        self.area_path = args.area_path
        self.chunk_size = args.chunk_size
        self.model_name = args.model_name

        # Get area images path dict
        self.camera_idx_paths = [os.path.join(args.area_path, cam_idx) 
                                for cam_idx in os.listdir(args.area_path)]
        self.image_paths = {}
        for i in len(self.camera_idx_paths):
            single_cam_img_paths = [os.path.join(self.camera_idx_paths[i], img_fp) 
                                    for img_fp in os.listdir(self.camera_idx_paths[i])]
            self.image_paths[self.camera_idx_paths[i]] = single_cam_img_paths
            single_cam_img_paths = []
        
        

        # intialize model
        if model_name == 'depthanything3':

        

    def __model_inference(self, image_paths):
        """
        The number of image frames that need to be input for 
        each inference is determined by the chunk_size parameter
        Args:
            image_paths(Dict[str: List[str,...]]): all images paths
        """
        chunk_frames = []
        for k, v in image_paths.items():
            chunk_frames = image_paths[k][]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FF3DR')
    parser.add_argument('--area_path', type=str, default='./dataset/WHU-OMVS/train/area1', required=True, help='Area images for ff3dr reconstruction')
    parser.add_argument('--config_path', type=str, default='./configs/da3_config.yml', required=False, help='Stored model hyper-parameters')
    parser.add_argument('--output_path', type=str, default='./exp', required=False, help='Save outputs')
    parser.add_argument('--model_name', type=str, default='depthanything3', required=False, help='depthanything3, mapanything, pi3 and vggt model')
    parser.add_argument('--chunk_size', type=int, default=30, required=False, help='')
    args = parser.parse_args

    ff3dr = FF3DR(args.area_path)

