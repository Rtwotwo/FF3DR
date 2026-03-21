import os
import torch
from pathlib import Path
import logging
import argparse

# TODO: address the outfolder import issue
from metrics import compute_metrics
from utils.config_utils import load_config
from models.depthanything3.api import DepthAnything3
from models.mapanything.models.mapanything import MapAnything
from models.pi3.models.pi3 import PI3
from models.vggt.models.vggt import VGGT
logger = logging.getLogger(__name__)


class FF3DR_MetricInfer():
        """
        Use pretrained model to infer dataset to get metric results.
        Args:
            cfg(dict): store all configurations for inference.
        """
        def __init__(self, cfg):
            self.cfg = cfg
            if torch.cuda.is_available():
                  self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >=8 else torch.float16
            else:
                  self.dtype = torch.float32
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model_name = self.cfg['Model']
            


            






