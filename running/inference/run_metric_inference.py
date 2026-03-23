import os
import torch
import logging
import argparse
import json
from pathlib import Path
from safetensors.torch import load_file

# TODO: address the outfolder import issue
from metrics import compute_metrics
from utils.config_utils import load_config
from uniception.models.utils.transformer_blocks import Mlp
from models.depthanything3.api import DepthAnything3
from models.mapanything.models.mapanything import MapAnything
from models.pi3.models.pi3 import Pi3
from models.vggt.models.vggt import VGGT
logger = logging.getLogger(__name__)


class MetricInfer:
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
            self.model_name = self.cfg['Model']['name']

        def _load_model(self,):
            """
            load pretrained model for inference, contraining
            depthanything3, mapanything, pi3 and vggt
            """  
            if self.model_name == "depthanything3":
                da3_config_path = self._cfg_get_weight("depthanything3", "DA3_CONFIG", fallback_flat_key="DA3_CONFIG")
                da3_weight_path = self._cfg_get_weight("depthanything3", "DA3", fallback_flat_key="DA3")
                with open(da3_config_path, "r") as f:
                    da3_config = json.load(f)
                model = DepthAnything3(**da3_config)
                state_dict = load_file(da3_weight_path)
                model.load_state_dict(state_dict, strict=False)
            elif self.model_name == "mapanything":
                try:
                    with open(self.config["Weights"]["mapanything"]["MAP_CONFIG"], "r") as f:
                        map_config = json.load(f)
                    # Compat fix: newer uniception expects callable mlp_layer instead of string.
                    if isinstance(map_config, dict):
                        info_cfg = map_config.get("info_sharing_config", {})
                        module_args = info_cfg.get("module_args", {})
                        if module_args.get("mlp_layer", None) == "mlp":
                            module_args["mlp_layer"] = Mlp
                    model = MapAnything(**map_config)
                    state_dict = load_file(self.config["Weights"]["mapanything"]["MAP"])
                    model.load_state_dict(state_dict, strict=False)
                except Exception:
                    model = MapAnything.from_pretrained(self.config["Weights"]["mapanything"]["MAP_URL"])
            elif self.model_name == "pi3":
                _ = self.config["Weights"]["pi3"]["PI3_CONFIG"]
                model = Pi3()
                state_dict = load_file(self.config["Weights"]["pi3"]["PI3"])
                model.load_state_dict(state_dict, strict=False)
            elif self.model_name == "vggt":
                model = VGGT()
                state_dict = torch.load(self.config["Weights"]["vggt"]["VGGT"], map_location=self.device)
                if isinstance(state_dict, dict):
                    model.load_state_dict(state_dict, strict=False)
            else:
                raise RuntimeError("[ERROR] model_name must be one of depthanything3/mapanything/pi3/vggt")
            model.eval().to(self.device)
            logger.info("[INFO] Model loaded: %s", self.model_name)
            return model
        


                






