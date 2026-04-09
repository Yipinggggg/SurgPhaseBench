import torch
import torchvision
from .MetaFormer import MetaFormerFeature
from .convnextv2 import ConvNextFeature
from .convnextv2 import ConvNextFeatureLast
from .pvtv2 import PVTV2Feature
from transformers import AutoModel
# from .surgenet_dinov3 import SurgeNet_DINOv3_Feature
# from .surgenet_dinov3 import DINOv3_vitb_Feature

urls = {"convnextv2_tiny_imagenet1k": 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt',
		"SurgeNet-ConvNextv2": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNet_ConvNextv2_checkpoint_epoch0050_teacher.pth?download=true",
		'pvtv2_b2_imagenet1k': 'https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth',
		"SurgeNet-PVTv2": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNet_PVTv2_checkpoint_epoch0050_teacher.pth?download=true",
		"SurgeNetSmall": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNetSmall_checkpoint_epoch0050_teacher.pth?download=true",
		"SurgeNet": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNet_checkpoint_epoch0050_teacher.pth?download=true",
		"SurgeNet-RAMIE": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/RAMIE_checkpoint_epoch0050_teacher.pth?download=true",
		"SurgeNet-Public": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/Public_checkpoint_epoch0050_teacher.pth?download=true",
		"SurgeNet-Cholec": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/CHOLEC_checkpoint_epoch0050_teacher.pth?download=true",
}


def metaformer(pretrained=True,**kwargs):
	weights = None
	if pretrained:
		model = MetaFormerFeature(pretrained='ImageNet')
		model.cuda()
	else:
		raise NotImplementedError("Only pretrained model supported")
	return model

def metaformer_in21k(pretrained=True,**kwargs):
	weights = None
	if pretrained:
		model = MetaFormerFeature(pretrained='ImageNet21k')
		model.cuda()
	else:
		raise NotImplementedError("Only pretrained model supported")
	return model

def surgenet_Cholec(pretrained=True,**kwargs):
	if pretrained:
		weights=urls["SurgeNet-Cholec"]
		model = MetaFormerFeature(pretrained='SurgNet', weights=weights)
		model.cuda()
	else:
		raise NotImplementedError("Only pretrained model supported")
	return model

def surgenet_public(pretrained=True,**kwargs): 
	if pretrained:
		weights=urls["SurgeNet-Public"]
		model = MetaFormerFeature(pretrained='SurgNet', weights=weights)
		model.cuda()
	else:
		raise NotImplementedError("Only pretrained model supported")
	
	return model

def surgnet_small(pretrained=True,**kwargs): 
	if pretrained:
		weights=urls["SurgeNetSmall"]
		model = MetaFormerFeature(pretrained='SurgNet', weights=weights)
		model.cuda()
	else:
		raise NotImplementedError("Only pretrained model supported")
	
	return model

def surgnet_big(pretrained=True,**kwargs): 
	if pretrained:
		weights=urls["SurgeNet"]
		model = MetaFormerFeature(pretrained='SurgNet', weights=weights)
		model.cuda()
	else:
		raise NotImplementedError("Only pretrained model supported")
	
	return model

def surgnet_RAMIE(pretrained=True,**kwargs): 
	if pretrained:
		weights=urls["SurgeNet-RAMIE"]
		model = MetaFormerFeature(pretrained='SurgNet', weights=weights)
		model.cuda()
	else:
		raise NotImplementedError("Only pretrained model supported")
	
	return model

def surgnet_XL(pretrained=True,**kwargs): 
	if pretrained:
		weight_path='/projects/prjs0797/Yiping/SurgicalPhaseRecognition/SurgeNet_Phase/train_scripts/SurgeNet/SurgeNetXL_checkpoint_epoch0050_teacher.pth'
		weights=torch.load(weight_path, weights_only=False)
		model = MetaFormerFeature(pretrained='SurgNet', weights=weights)
		model.cuda()
	else:
		raise NotImplementedError("Only pretrained model supported")
	
	return model

def convnextv2(pretrained=True,**kwargs): 
	if pretrained:
		model = ConvNextFeature(pretrained_weights=urls["convnextv2_tiny_imagenet1k"])
		model.cuda()
	else:
		model = ConvNextFeature(pretrained_weights=None)
		model.cuda()
	
	return model

def pvtv2(pretrained=True,**kwargs): 
	if pretrained:
		model = PVTV2Feature(pretrained_weights=urls["pvtv2_b2_imagenet1k"])
		model.cuda()
	else:
		model = PVTV2Feature(pretrained_weights=None)
		model.cuda()
	
	return model

def surgenet_ConvNextv2(pretrained=True,**kwargs): 
	if pretrained:
		model = ConvNextFeature(pretrained_weights=urls["SurgeNet-ConvNextv2"])
		model.cuda()
	else:
		raise NotImplementedError("Only pretrained model supported")
	
	return model

def surgenet_ConvNextv2Last(pretrained=True,**kwargs): 
	if pretrained:
		model = ConvNextFeatureLast(pretrained_weights=urls["SurgeNet-ConvNextv2"])
		model.cuda()
	else:
		raise NotImplementedError("Only pretrained model supported")
	
	return model

def surgenet_PVTv2(pretrained=True,**kwargs): 
	if pretrained:
		model = PVTV2Feature(pretrained_weights=urls["SurgeNet-PVTv2"])
		model.cuda()
	else:
		raise NotImplementedError("Only pretrained model supported")
	
	return model

def surgenet_public(pretrained=True,**kwargs): 
	if pretrained:
		model = MetaFormerFeature(pretrained='SurgNet', weights=urls["SurgeNet-Public"])
		model.cuda()
	else:
		raise NotImplementedError("Only pretrained model supported")
	
	return model

# def surgenet_vitb_dinov3(pretrained=True,**kwargs): 
# 	if pretrained:
# 		model = SurgeNet_DINOv3_vitb_Feature(pretrained=True)
# 		model.cuda()
# 	else:
# 		raise NotImplementedError("Only pretrained model supported")
# 	return model


# def vitb_dinov3(pretrained=True,**kwargs):
# 	if pretrained:
# 		model = DINOv3_vitb_Feature(pretrained=True)
# 		model.cuda()
# 	else:
# 		raise NotImplementedError("Only pretrained model supported")
# 	return model