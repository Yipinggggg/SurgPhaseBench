import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from timm.models.vision_transformer import VisionTransformer
from functools import partial
from torch import nn
from huggingface_hub import snapshot_download

def endovit(pretrained=True, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)).eval()
    model = model.to('cuda')
    if pretrained:
        weight_path = "/projects/prjs0797/Yiping/SurgicalPhaseRecognition/SurgeNet_Phase/train_scripts/EndoViT/endovit_SPR.pth"
        model_weights = torch.load(weight_path, weights_only=False)['model']
        loading = model.load_state_dict(model_weights, strict=False)
        print(loading)
    else:
        raise NotImplementedError("Only pretrained model supported")
    
    return model