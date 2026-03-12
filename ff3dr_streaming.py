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
import logging
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
    frame of images from 5 cameras to FF3DR reconstruction.
    Args:
        args: The registered user parameters can be set by the user.
    """
    def __init__(self, args):
        # Passing hyper-params for FF3DR 
        self.area_path = args.area_path
        self.chunk_size = args.chunk_size
        self.model_name = args.model_name
        self.device = args.device

        # Get area images path dict
        self.camera_idx_paths = [os.path.join(args.area_path, cam_idx) 
                                for cam_idx in os.listdir(args.area_path)]
        self.image_paths = {}
        for i in range(len(self.camera_idx_paths)):
            single_cam_img_paths = [os.path.join(self.camera_idx_paths[i], img_fp) 
                                    for img_fp in os.listdir(self.camera_idx_paths[i])]
            self.image_paths[self.camera_idx_paths[i]] = single_cam_img_paths
            single_cam_img_paths = []
        
        # intialize model
        self.config = load_config(args.config_path)
        if args.model_name == 'depthanything3':
            # load depthanything config and state_dict
            with open(self.config["Weights"]["depthanything3"]["DA3_CONFIG"]) as f:
                da3_config = json.load(f)
            self.model = DepthAnything3(**da3_config)
            state_dict = load_file(self.config["Weights"]["depthanything3"]["DA3"])
            self.model.load_state_dict(state_dict, strict=False)
        elif args.model_name == 'mapanything':
            # load mapanything config and state_dict
            with open(self.config["Weights"]["mapanything"]["MAP_CONFIG"]) as f:
                map_config = json.load(f)
            self.model = MapAnything()
            state_dict = load_file(self.config["Weights"]["mapanything"]["MAP"])
            self.model.load_state_dict(state_dict, strict=False)
        elif args.model_name == 'pi3':
            # load pi3 config and state_dict
            with open(self.config['Weights']['pi3']['PI3_CONFIG']) as f:
                pi3_config = json.load(f)
            self.model = Pi3()
            state_dict = load_file(self.config['Weights']['pi3']['PI3'])
            self.model.load_state_dict(state_dict, strict=False)
        elif args.model_name == 'vggt':
            # load vggt state_dict
            self.model = VGGT()
            state_dict = torch.load(self.config['Weights']['vggt']['VGGT'], map_location=self.device)
        print(self.model)
        self.model.eval().to(self.device)

    def inference(self):
        """
        The number of image frames that need to be input for 
        each inference is determined by the chunk_size parameter
        """






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FF3DR')
    parser.add_argument('--area_path', type=str, default='./dataset/WHU-OMVS/train/area1/images', required=False, help='Area images for ff3dr reconstruction')
    parser.add_argument('--config_path', type=str, default='./configs/base_config.yml', required=False, help='Stored model hyper-parameters')
    parser.add_argument('--output_path', type=str, default='./exp', required=False, help='Save outputs')
    parser.add_argument('--model_name', type=str, default='mapanything', required=False, help='depthanything3, mapanything, pi3 and vggt model')
    parser.add_argument('--chunk_size', type=int, default=30, required=False, help='inference number of images once time')
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), required=False, help='computing dl device')
    args = parser.parse_args()

    ff3dr = FF3DR(args)

