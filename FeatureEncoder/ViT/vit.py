import torch
import torchvision

def vit_b(pretrained=True, **kwargs):
    if pretrained:
        model = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
    else:
        model = torchvision.models.vit_b_16()
    return model