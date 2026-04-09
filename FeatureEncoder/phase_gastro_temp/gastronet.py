import torch
import torchvision
from .ResNet import ResNet50Feature
from .ViT import vit_smallFeature

def RN50_GastroNet(pretrained=True,**kwargs):
	weights = None
	if pretrained:
		weights_path = "/home/20235694/SurgeNet_Phase/train_scripts/phase_gastro_temp/RN50_GastroNet-5M_DINOv1.pth"
		model = ResNet50Feature(channels=3, pretrained=None, url='').cuda()
		weights = torch.load(weights_path)
	else:
		raise NotImplementedError("Only pretrained model supported")
	return model


def VITS_GastroNet(pretrained=True,**kwargs):
	weights = None
	if pretrained:
		weights_path = "/home/20235694/SurgeNet_Phase/train_scripts/phase_gastro_temp/VITS_GastroNet-5M_DINOv1.pth"
		model = vit_smallFeature().cuda()
		weights = torch.load(weights_path)
	else:
		raise NotImplementedError("Only pretrained model supported")
	return model