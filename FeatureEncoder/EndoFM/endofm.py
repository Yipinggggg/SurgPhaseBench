import argparse
import json
import os
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from torch import nn
from tqdm import tqdm
import numpy as np

from sklearn.metrics import f1_score

# from datasets import UCF101, HMDB51, Kinetics
from .models import get_vit_base_patch16_224, get_aux_token_vit, SwinTransformer3D
from .utils import utils
from .utils.meters import TestMeter
from .utils.parser import load_config

# def endofm(pretrained=True,**kwargs):
#     endofm_parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
#     endofm_parser.add_argument("--cfg", dest="cfg_file", help="Path to the config file", type=str,
#                         default="models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml")
#     endofm_parser.add_argument("--opts", help="See utils/defaults.py for all options", default=None, nargs=argparse.REMAINDER)
#     endofm_args = endofm_parser.parse_args()
#     endofm_config = load_config(endofm_args)
#     model = get_vit_base_patch16_224(cfg=endofm_config, no_head=True)
#     print(model)
#     if pretrained:
#         ckpt = torch.load('checkpoints/endo_fm.pth', map_location='cpu')
#         if "teacher" in ckpt:
#             ckpt = ckpt["teacher"]
#         if "model_state" in ckpt:
#             ckpt = ckpt["model_state"]
#         if 'TimeSformer' in 'checkpoints/endo_fm.pth':
#             ckpt = {"backbone." + key[len("model."):]: value for key, value in ckpt.items()}
#             print(ckpt.keys())
#         renamed_checkpoint = {x[len("backbone."):]: y for x, y in ckpt.items() if x.startswith("backbone.")}
#         msg = model.load_state_dict(renamed_checkpoint, strict=False)
#         print(f"Loaded model with msg: {msg}")
#         model.cuda()
#     else:
#         raise NotImplementedError("Only pretrained model supported")
    
#     return model

def endofm(pretrained=True, cfg_file=None, opts=None, **kwargs):
    if cfg_file is None:
        cfg_file = "/projects/prjs0797/Yiping/SurgicalPhaseRecognition/SurgeNet_Phase/train_scripts/EndoFM/models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml"
    
    # Create a simple object to mimic the structure of parsed arguments
    class Args:
        pass
    
    endofm_args = Args()
    endofm_args.cfg_file = cfg_file
    endofm_args.opts = opts if opts is not None else []

    endofm_config = load_config(endofm_args)
    model = get_vit_base_patch16_224(cfg=endofm_config, no_head=True)
    print(model)

    if pretrained:
        ckpt = torch.load('/projects/prjs0797/Yiping/SurgicalPhaseRecognition/SurgeNet_Phase/train_scripts/EndoFM/checkpoints/endo_fm.pth', map_location='cpu', weights_only=False)
        if "teacher" in ckpt:
            ckpt = ckpt["teacher"]
        if "model_state" in ckpt:
            ckpt = ckpt["model_state"]
        if 'TimeSformer' in 'checkpoints/endo_fm.pth':
            ckpt = {"backbone." + key[len("model."):]: value for key, value in ckpt.items()}
            print(ckpt.keys())
        renamed_checkpoint = {x[len("backbone."):]: y for x, y in ckpt.items() if x.startswith("backbone.")}
        msg = model.load_state_dict(renamed_checkpoint, strict=False)
        print(f"Loaded model with msg: {msg}")
        model.cuda()
    else:
        raise NotImplementedError("Only pretrained model supported")
    
    return model