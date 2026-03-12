import os
import torch


class Cameras:
    """
    Caneras information to register parameters
    Args:
        
    """
    CAMERA_ID = None
    WIDTH = None
    HEIGHT = None
    PIXELSIZE = None
    PARAMS = [] # [fx,fy,cx,cy]
    DISTORTION = [] # [K1, K2, K3, P1, P2]


class Images:
    """
    Images information to register parameters
    Args:
        
    """
    IMAGE_ID = None
    CAMERA_ID = None
    Rwc = [] # 3x3
    twc = [] # 1x3
    MINDEPTH = None
    MAXDEPTH = None
    NAME = None


class 