import os
import torch
from pathlib import Path
import logging
from models.vggt.models.vggt import VGGT
logger = logging.getLogger(__name__)


class VGGTInference:
    """
    Adapting VGGT model for inference WHU-OMVS prediction.
    """
    def __init__(self, state_dict_path:str,
                 device='cuda'):
        # intialize model and load state dict
        self.model = VGGT()
        if state_dict_path is not None:
            logger.error(f"[ERROR] state dict path is not provided, cannot load model.")
        state_dict = torch.load(state_dict_path, map_location="cuda")
        self.model.load_state_dict(state_dict)
        self.model.eval().to(device)

    def 



