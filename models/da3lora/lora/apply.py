from __future__ import annotations

import re
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from da3lora.lora.lora_layer import LoRALinear, LoRAConv2d


LORA_TARGET_LINEAR_PATTERNS = [
    r"backbone\.pretrained\.blocks\.\d+\.attn\.qkv$",
    r"backbone\.pretrained\.blocks\.\d+\.attn\.proj$",
    r"backbone\.pretrained\.blocks\.\d+\.mlp\.fc1$",
    r"backbone\.pretrained\.blocks\.\d+\.mlp\.fc2$",
    r"head\.scratch\.refinenet\d+\.resConfUnit\d+\.conv1\.conv$",
    r"head\.scratch\.refinenet\d+\.resConfUnit\d+\.conv2\.conv$",
    r"cam_dec\.trunk\.\d+\.attn\.qkv$",
    r"cam_dec\.trunk\.\d+\.attn\.proj$",
    r"cam_dec\.trunk\.\d+\.mlp\.fc1$",
    r"cam_dec\.trunk\.\d+\.mlp\.fc2$",
]

LORA_TARGET_CONV_PATTERNS = [
    r"head\.projects\.\d+$",
    r"head\.scratch\.layer\d+_rnn$",
    r"head\.scratch\.refinenet\d+\.resConfUnit\d+\.conv1\.conv$",
    r"head\.scratch\.refinenet\d+\.resConfUnit\d+\.conv2\.conv$",
]


def _match_patterns(name, patterns):
    for pat in patterns:
        if re.fullmatch(pat, name):
            return True
    return False


def apply_lora_to_model(
    model,
    r=8,
    lora_alpha=16.0,
    dropout=0.0,
    target_linear_patterns=None,
    target_conv_patterns=None,
    verbose=True,
):
    if target_linear_patterns is None:
        target_linear_patterns = LORA_TARGET_LINEAR_PATTERNS
    if target_conv_patterns is None:
        target_conv_patterns = LORA_TARGET_CONV_PATTERNS

    linear_count = 0
    conv_count = 0

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _match_patterns(name, target_linear_patterns):
            lora_layer = LoRALinear(
                module, r=r, lora_alpha=lora_alpha, dropout=dropout,
            )
            _set_nested_attr(model, name, lora_layer)
            linear_count += 1
            if verbose:
                print("[LoRA] Linear: {} (in={}, out={})".format(
                    name, module.in_features, module.out_features))

        elif isinstance(module, nn.Conv2d) and _match_patterns(name, target_conv_patterns):
            lora_layer = LoRAConv2d(
                module, r=r, lora_alpha=lora_alpha, dropout=dropout,
            )
            _set_nested_attr(model, name, lora_layer)
            conv_count += 1
            if verbose:
                print("[LoRA] Conv2d: {} (in={}, out={})".format(
                    name, module.in_channels, module.out_channels))

    if verbose:
        total = linear_count + conv_count
        print("[LoRA] Applied to {} layers ({} Linear, {} Conv2d), rank={}, alpha={}".format(
            total, linear_count, conv_count, r, lora_alpha))

    return model


def freeze_non_lora_params(model, verbose=True):
    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()

    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            if hasattr(module, "weight"):
                module.weight.requires_grad_(False)
            if hasattr(module, "bias"):
                module.bias.requires_grad_(False)

    if verbose:
        total = trainable + frozen
        print("[LoRA] Frozen {:,} params, trainable {:,} params ({:.2f}% of total)".format(
            frozen, trainable, 100 * trainable / total))


def get_lora_params(model):
    lora_a_params = []
    lora_b_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" in name:
            lora_a_params.append(param)
        elif "lora_B" in name:
            lora_b_params.append(param)

    return {
        "lora_A": lora_a_params,
        "lora_B": lora_b_params,
        "all": lora_a_params + lora_b_params,
    }


def save_lora_weights(model, save_path):
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state_dict[name] = param.data.cpu()

    torch.save(lora_state_dict, save_path)
    print("[LoRA] Saved {} adapter tensors to {}".format(len(lora_state_dict), save_path))


def load_lora_weights(model, load_path, strict=True):
    lora_state_dict = torch.load(load_path, map_location="cpu")
    model_state_dict = model.state_dict()

    loaded = 0
    for name, param in lora_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name].copy_(param)
            loaded += 1
        elif strict:
            raise KeyError("LoRA weight '{}' not found in model".format(name))

    print("[LoRA] Loaded {}/{} adapter tensors from {}".format(
        loaded, len(lora_state_dict), load_path))


def merge_lora_weights(model):
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            module.merge()
    print("[LoRA] All adapters merged into base weights")


def unmerge_lora_weights(model):
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            module.unmerge()
    print("[LoRA] All adapters unmerged from base weights")


def _set_nested_attr(model, name, new_module):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)
