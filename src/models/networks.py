import torch
from torch import nn
import torchvision
import math
import timm
import importlib
import sys
from pathlib import Path
from functools import lru_cache


# Ensure FeatureEncoder packages are importable regardless of entry-point cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_repo_root_on_path():
	if str(_REPO_ROOT) not in sys.path:
		sys.path.insert(0, str(_REPO_ROOT))


@lru_cache(maxsize=None)
def _import_feature_encoder_module(module_name):
	_ensure_repo_root_on_path()
	return importlib.import_module(f"FeatureEncoder.{module_name}")


@lru_cache(maxsize=1)
def _import_vit_module():
	_ensure_repo_root_on_path()
	return importlib.import_module("FeatureEncoder.ViT.vit")


class TemporalCNN(nn.Module):

	def __init__(self,out_size,backbone,head,opts):

		super(TemporalCNN, self).__init__()
		self.backbone = backbone
		self.cnn = CNN(out_size,backbone,opts)
		if head == 'lstm':
			self.temporal_head = LSTMHead(self.cnn.feature_size,out_size,opts.seq_len)

	def forward(self,x):

		x = self.extract_image_features(x)
		x = self.temporal_head(x)

		return x

	def forward_sliding_window(self,x):

		x = self.extract_image_features(x)
		x = self.temporal_head.forward_sliding_window(x)

		return x

	def extract_image_features(self,x):

		B = x.size(0)
		S = x.size(1)
		x = x.flatten(end_dim=1)
		if self.backbone == 'endofm':
			# target shape: (batchsize, 3, 1, height, width)
			x = x.unsqueeze(2)
		if self.backbone == 'endovit':
			x = self.cnn.featureNet.forward_features(x)
			# only take the class token
			x = x[:,0,:]
		else:
			x = self.cnn.featureNet(x)
		x = x.view(B,S,-1)

		return x

class LSTMHead(nn.Module):

	def __init__(self,feature_size,out_size,train_len,lstm_size=512):

		super(LSTMHead, self).__init__()

		self.lstm = nn.LSTM(feature_size,lstm_size,batch_first=True)
		self.out_layer = nn.Linear(lstm_size,out_size)

		self.train_len = train_len

		self.hidden_state = None
		self.prev_feat = None

	def forward(self,x):

		x, hidden_state = self.lstm(x,self.hidden_state)
		x = self.out_layer(x)

		self.hidden_state = tuple(h.detach() for h in hidden_state)

		return [x]

	def forward_sliding_window(self,x):

		#print('#')
		if self.prev_feat is not None:
			x_sliding = torch.cat((self.prev_feat,x),dim=1)
		else:
			x_sliding = x

		x_sliding = torch.cat([
			x_sliding[:,i:i+self.train_len,:] for i in range(x_sliding.size(1)-self.train_len+1)
		])
		x_sliding, _ = self.lstm(x_sliding)
		x_sliding = self.out_layer(x_sliding)

		if self.prev_feat is not None:
			#_,pred = x_sliding.max(dim=-1)
			#print(pred)
			x_sliding = x_sliding[:,-1,:].unsqueeze(dim=0)
			#_,pred = x_sliding.max(dim=-1)
			#print(pred)
		else:
			first_preds = x_sliding[0,:-1,:].unsqueeze(dim=0)
			x_sliding = x_sliding[:,-1,:].unsqueeze(dim=0)
			x_sliding = torch.cat((first_preds,x_sliding),dim=1)

		self.prev_feat = x[:,1-self.train_len:,:].detach()
		return [x_sliding]

	def reset(self):

		self.hidden_state = None
		self.prev_feat = None


class CNN(nn.Module):

	def __init__(self,out_size,backbone,opts):

		super(CNN, self).__init__()
		self.backbone = backbone

		if backbone == 'alexnet':
			self.featureNet = torchvision.models.alexnet(pretrained=True)
			self.featureNet.classifier = self.featureNet.classifier[:3]
			self.feature_size = 4096
		elif backbone == 'vgg11':
			self.featureNet = torchvision.models.vgg11(pretrained=True)
			self.featureNet.classifier = self.featureNet.classifier[:3]
			self.feature_size = 4096
		elif backbone == 'vgg16':
			self.featureNet = torchvision.models.vgg16(pretrained=True)
			self.featureNet.classifier = self.featureNet.classifier[:3]
			self.feature_size = 4096
		elif backbone == 'resnet18':
			self.featureNet = torchvision.models.resnet18(pretrained=True)
			self.featureNet.fc = Identity()
			self.feature_size = 512
		elif backbone == 'resnet34':
			self.featureNet = torchvision.models.resnet34(pretrained=True)
			self.featureNet.fc = Identity()
			self.feature_size = 512
		elif backbone == 'resnet50':
			# https://docs.pytorch.org/vision/stable/models.html
			from torchvision.models import ResNet50_Weights
			self.featureNet = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
			self.featureNet.fc = Identity()
			self.feature_size = 2048
			if opts.freeze:
				for param in self.featureNet.parameters():
					param.requires_grad = False
		elif backbone == 'resnet18_gn':
			self.featureNet = _import_feature_encoder_module('group_norm').resnet18_gn(pretrained=True)
			self.featureNet.fc = Identity()
			self.feature_size = 512
		elif backbone == 'resnet34_gn':
			self.featureNet = _import_feature_encoder_module('group_norm').resnet34_gn(pretrained=True)
			self.featureNet.fc = Identity()
			self.feature_size = 512
		elif backbone == 'resnet50_gn':
			self.featureNet = _import_feature_encoder_module('group_norm').resnet50_gn(pretrained=True)
			self.featureNet.fc = Identity()
			self.feature_size = 2048
			if opts.freeze:
				for param in self.featureNet.parameters():
					param.requires_grad = False
				for param in self.featureNet.layer4.parameters():
					param.requires_grad = True
		elif backbone == 'convnext':
			self.featureNet = _import_feature_encoder_module('convnext').convnext_tiny(pretrained=True)
			self.featureNet.head = Identity()
			self.feature_size = 768
			if opts.freeze:
				for i in [0,1,2]:
					for param in self.featureNet.downsample_layers[i].parameters():
						param.requires_grad = False
					for param in self.featureNet.stages[i].parameters():
						param.requires_grad = False
		elif backbone == 'nfnet':
			self.featureNet = timm.create_model('nfnet_l0', pretrained=True)
			#self.featureNet.head.fc = Identity()
			self.featureNet.head.fc = nn.Linear(in_features=2304, out_features=4096, bias=True)
			self.feature_size = 4096
			# TODO: test if the FC really influences performances
			# NOTES ON NFNET: can only get acceptable results with following hyperparams: BS:24, LR:1e-4, lossX3 (possibly L2:2e-5, w/ CLS)

		elif backbone == 'surgenet_big':
			self.featureNet = _import_feature_encoder_module('SurgeNet').surgnet_big(pretrained=True)
			self.feature_size = 1024
			if opts.freeze:
				# print the params in the network
				print(self.featureNet)
				for param in self.featureNet.parameters():
					param.requires_grad = False
				for param in self.featureNet.metaformer.stages[3].parameters():
					param.requires_grad = True
				for param in self.featureNet.metaformer.norm.parameters():
					param.requires_grad = True
		elif backbone == 'surgenet_small':
			self.featureNet = _import_feature_encoder_module('SurgeNet').surgnet_small(pretrained=True)
			self.feature_size = 1024
			if opts.freeze:
				# print the params in the network
				print(self.featureNet)
				for param in self.featureNet.parameters():
					param.requires_grad = False
				for param in self.featureNet.metaformer.stages[3].parameters():
					param.requires_grad = True
				for param in self.featureNet.metaformer.norm.parameters():
					param.requires_grad = True
		elif backbone == 'surgenet_RAMIE':
			self.featureNet = _import_feature_encoder_module('SurgeNet').surgnet_RAMIE(pretrained=True)
			self.feature_size = 1024
			if opts.freeze:
				# print the params in the network
				print(self.featureNet)
				for param in self.featureNet.parameters():
					param.requires_grad = False
				for param in self.featureNet.metaformer.stages[3].parameters():
					param.requires_grad = True
				for param in self.featureNet.metaformer.norm.parameters():
					param.requires_grad = True
		elif backbone == 'surgenet_XL':
			self.featureNet = _import_feature_encoder_module('SurgeNet').surgnet_XL(pretrained=True)
			self.feature_size = 1024
			if opts.freeze:
				# print the params in the network
				print(self.featureNet)
				for param in self.featureNet.parameters():
					param.requires_grad = False
				for param in self.featureNet.metaformer.stages[3].parameters():
					param.requires_grad = True
				for param in self.featureNet.metaformer.norm.parameters():
					param.requires_grad = True
		elif backbone == 'surgenet_public':
			self.featureNet = _import_feature_encoder_module('SurgeNet').surgenet_public(pretrained=True)
			self.feature_size = 1024
			if opts.freeze:
				# print the params in the network
				print(self.featureNet)
				for param in self.featureNet.parameters():
					param.requires_grad = False
				for param in self.featureNet.metaformer.stages[3].parameters():
					param.requires_grad = True
				for param in self.featureNet.metaformer.norm.parameters():
					param.requires_grad = True
		elif backbone == 'surgenet_Cholec':
			self.featureNet = _import_feature_encoder_module('SurgeNet').surgenet_Cholec(pretrained=True)
			self.feature_size = 1024
			if opts.freeze:
				# print the params in the network
				print(self.featureNet)
				for param in self.featureNet.parameters():
					param.requires_grad = False
				for param in self.featureNet.metaformer.stages[3].parameters():
					param.requires_grad = True
				for param in self.featureNet.metaformer.norm.parameters():
					param.requires_grad = True
		elif backbone == 'metaformer':
			self.featureNet = _import_feature_encoder_module('SurgeNet').metaformer(pretrained=True)
			self.feature_size = 1024
			if opts.freeze:
				# print the params in the network
				print(self.featureNet)
				for param in self.featureNet.parameters():
					param.requires_grad = False
				for param in self.featureNet.metaformer.stages[3].parameters():
					param.requires_grad = True
				for param in self.featureNet.metaformer.norm.parameters():
					param.requires_grad = True
		elif backbone == 'metaformer_in21k':
			self.featureNet = _import_feature_encoder_module('SurgeNet').metaformer_in21k(pretrained=True)
			self.feature_size = 1024
			if opts.freeze:
				# print the params in the network
				print(self.featureNet)
				for param in self.featureNet.parameters():
					param.requires_grad = False
				for param in self.featureNet.metaformer.stages[3].parameters():
					param.requires_grad = True
				for param in self.featureNet.metaformer.norm.parameters():
					param.requires_grad = True
	
		elif backbone == 'convnextv2':
			self.featureNet = _import_feature_encoder_module('SurgeNet').convnextv2(pretrained=True)
			self.feature_size = 1440
			if opts.freeze:
				# print the params in the network
				print(self.featureNet)
				# freeze everything
				for param in self.featureNet.parameters():
					param.requires_grad = False
				for param in self.featureNet.convnext.stages[3].parameters():
					param.requires_grad = True
				for param in self.featureNet.convnext.norm.parameters():
					param.requires_grad = True
	
		elif backbone == 'surgenet_ConvNextv2':
			self.featureNet = _import_feature_encoder_module('SurgeNet').surgenet_ConvNextv2(pretrained=True)
			self.feature_size = 1440
			if opts.freeze:
				# print the params in the network
				print(self.featureNet)
				# freeze everything
				for param in self.featureNet.parameters():
					param.requires_grad = False
				for param in self.featureNet.convnext.stages[3].parameters():
					param.requires_grad = True
				for param in self.featureNet.convnext.norm.parameters():
					param.requires_grad = True
		
		elif backbone == 'surgenet_ConvNextv2Last':
			self.featureNet = _import_feature_encoder_module('SurgeNet').surgenet_ConvNextv2Last(pretrained=True)
			self.feature_size = 768
			if opts.freeze:
				# print the params in the network
				print(self.featureNet)
				# freeze everything
				for param in self.featureNet.parameters():
					param.requires_grad = False
				for param in self.featureNet.convnext.stages[3].parameters():
					param.requires_grad = True
				for param in self.featureNet.convnext.norm.parameters():
					param.requires_grad = True
		elif backbone == 'pvtv2':
			self.featureNet = _import_feature_encoder_module('SurgeNet').pvtv2(pretrained=True)
			self.feature_size = 1024
			if opts.freeze:
				# print the params in the network
				print(self.featureNet)
				# freeze everything
				for param in self.featureNet.parameters():
					param.requires_grad = False
				for param in self.featureNet.pvtv2.patch_embed4.parameters():
					param.requires_grad = True
				for param in self.featureNet.pvtv2.block4.parameters():
					param.requires_grad = True
				for param in self.featureNet.pvtv2.norm4.parameters():
					param.requires_grad = True
		elif backbone == 'surgenet_PVTv2':
			self.featureNet = _import_feature_encoder_module('SurgeNet').surgenet_PVTv2(pretrained=True)
			self.feature_size = 1024
			if opts.freeze:
				# print the params in the network
				print(self.featureNet)
				# freeze everything
				for param in self.featureNet.parameters():
					param.requires_grad = False
				for param in self.featureNet.pvtv2.patch_embed4.parameters():
					param.requires_grad = True
				for param in self.featureNet.pvtv2.block4.parameters():
					param.requires_grad = True
				for param in self.featureNet.pvtv2.norm4.parameters():
					param.requires_grad = True
		elif backbone == 'vitb_dinov3':
			self.featureNet = _import_feature_encoder_module('SurgeNet').DINOv3_vitb(pretrained=True)
			if opts.freeze:
				for name, param in self.featureNet.named_parameters():
					param.requires_grad = False
			self.feature_size = 768
		elif backbone == 'vitl_dinov3':
			self.featureNet = _import_feature_encoder_module('SurgeNet').DINOv3_vitl(pretrained=True)
			if opts.freeze:
				for name, param in self.featureNet.named_parameters():
					param.requires_grad = False
			self.feature_size = 1024
		elif backbone == 'vits_dinov3':
			self.featureNet = _import_feature_encoder_module('SurgeNet').DINOv3_vits(pretrained=True)
			if opts.freeze:
				for name, param in self.featureNet.named_parameters():
					param.requires_grad = False
			self.feature_size = 384
		elif backbone == 'vits_dinov2':
			self.featureNet = _import_feature_encoder_module('SurgeNet').DINOv2_vits(pretrained=True)
			if opts.freeze:
				for name, param in self.featureNet.named_parameters():
					param.requires_grad = False
			self.feature_size = 384
		elif backbone == 'vitb_dinov2':
			self.featureNet = _import_feature_encoder_module('SurgeNet').DINOv2_vitb(pretrained=True)
			if opts.freeze:
				for name, param in self.featureNet.named_parameters():
					param.requires_grad = False
			self.feature_size = 768
		elif backbone == 'vitl_dinov2':
			self.featureNet = _import_feature_encoder_module('SurgeNet').DINOv2_vitl(pretrained=True)
			if opts.freeze:
				for name, param in self.featureNet.named_parameters():
					param.requires_grad = False
			self.feature_size = 1024
		elif backbone == 'vitb_dinov1':
			self.featureNet = _import_feature_encoder_module('SurgeNet').DINOv1_vitb(pretrained=True)
			if opts.freeze:
				for name, param in self.featureNet.named_parameters():
					param.requires_grad = False
			self.feature_size = 768
		elif backbone == 'vits_dinov1':
			self.featureNet = _import_feature_encoder_module('SurgeNet').DINOv1_vits(pretrained=True)
			if opts.freeze:
				for name, param in self.featureNet.named_parameters():
					param.requires_grad = False
			self.feature_size = 384
		elif backbone == 'vits_dinov1_surgenet':
			self.featureNet = _import_feature_encoder_module('SurgeNet').SurgeNet_DINOv1_vits(pretrained=True)
			if opts.freeze:
				for name, param in self.featureNet.named_parameters():
					param.requires_grad = False
			self.feature_size = 384
		elif backbone == 'vitb_dinov1_surgenet':
			self.featureNet = _import_feature_encoder_module('SurgeNet').SurgeNet_DINOv1_vitb(pretrained=True)
			if opts.freeze:
				for name, param in self.featureNet.named_parameters():
					param.requires_grad = False
			self.feature_size = 768
		elif backbone == 'vits_dinov2_surgenet':
			self.featureNet = _import_feature_encoder_module('SurgeNet').SurgeNet_DINOv2_vits(pretrained=True)
			if opts.freeze:
				for name, param in self.featureNet.named_parameters():
					param.requires_grad = False
			self.feature_size = 384
		elif backbone == 'vanilla_vits':
			self.featureNet = _import_feature_encoder_module('SurgeNet').vanilla_vits(pretrained=False)
			if opts.freeze:
				for name, param in self.featureNet.named_parameters():
					param.requires_grad = False
			self.feature_size = 384
		elif backbone == 'vanilla_vitb':
			self.featureNet = _import_feature_encoder_module('SurgeNet').vanilla_vitb(pretrained=False)
			if opts.freeze:
				for name, param in self.featureNet.named_parameters():
					param.requires_grad = False
			self.feature_size = 768
		elif backbone == 'vanilla_vitl':
			self.featureNet = _import_feature_encoder_module('SurgeNet').vanilla_vitl(pretrained=False)
			if opts.freeze:
				for name, param in self.featureNet.named_parameters():
					param.requires_grad = False
			self.feature_size = 1024
		elif backbone == 'vitb_dinov2_surgenet':
			self.featureNet = _import_feature_encoder_module('SurgeNet').SurgeNet_DINOv2_vitb(pretrained=True)
			if opts.freeze:
				for name, param in self.featureNet.named_parameters():
					param.requires_grad = False
			self.feature_size = 768
		elif backbone == 'vitl_dinov2_surgenet':
			self.featureNet = _import_feature_encoder_module('SurgeNet').SurgeNet_DINOv2_vitl(pretrained=True)
			if opts.freeze:
				for name, param in self.featureNet.named_parameters():
					param.requires_grad = False
			self.feature_size = 1024
		elif backbone == 'vitl_dinov3_surgenet':
			self.featureNet = _import_feature_encoder_module('SurgeNet').SurgeNet_DINOv3_vitl(pretrained=True)
			if opts.freeze:
				for name, param in self.featureNet.named_parameters():
					param.requires_grad = False
			self.feature_size = 1024
		elif backbone == 'vitb_dinov3_surgenet':
			self.featureNet = _import_feature_encoder_module('SurgeNet').SurgeNet_DINOv3_vitb(pretrained=True)
			if opts.freeze:
				for name, param in self.featureNet.named_parameters():
					param.requires_grad = False
			self.feature_size = 768
		elif backbone == 'vits_dinov3_surgenet':
			self.featureNet = _import_feature_encoder_module('SurgeNet').SurgeNet_DINOv3_vits(pretrained=True)
			if opts.freeze:
				for name, param in self.featureNet.named_parameters():
					param.requires_grad = False
			self.feature_size = 384
		elif backbone == 'gastronet_rn50':
			self.featureNet = _import_feature_encoder_module('phase_gastro_temp').RN50_GastroNet(pretrained=True)
			self.feature_size = 2048
			if opts.freeze:
				# print the params in the network
				print(self.featureNet)
				# freeze everything
				for param in self.featureNet.parameters():
					param.requires_grad = False
				for param in self.featureNet.layer4.parameters():
					param.requires_grad = True
		elif backbone == 'gastronet_vits':
			self.featureNet = _import_feature_encoder_module('phase_gastro_temp').VITS_GastroNet(pretrained=True)
			self.feature_size = 384
			if opts.freeze:
				raise NotImplementedError("VITS_GastroNet does not support freezing yet")
		elif backbone == 'endofm':
			print("Using EndoFM backbone")
			self.featureNet = _import_feature_encoder_module('EndoFM').endofm(pretrained=True)
			self.feature_size = 768
			if opts.freeze:
				raise NotImplementedError("endofm does not support freezing yet")
		elif backbone == 'gsvit':
			self.featureNet = _import_feature_encoder_module('GSViT').GSViT(pretrained=True, batch_size=opts.batch_size)
			self.feature_size = 384
			if opts.freeze:
				raise NotImplementedError("VITS_GastroNet does not support freezing yet")
		elif backbone == 'endovit':
			self.featureNet = _import_feature_encoder_module('EndoViT').endovit(pretrained=True)
			self.feature_size = 768
			if opts.freeze:
				raise NotImplementedError("EndoViT does not support freezing yet")
		elif backbone == 'vit_b':
			self.featureNet = _import_vit_module().vit_b(pretrained=True)
			self.feature_size = 768
			if opts.freeze:
				raise NotImplementedError("ViT does not support freezing yet")
		else:
			raise NotImplementedError('Backbone not implemented!')
		self.out_layer = nn.Linear(self.feature_size,out_size)

	def forward(self,x):

		B = x.size(0)
		S = x.size(1)
		x = x.flatten(end_dim=1)
		H = x.size(2)
		W = x.size(3)
		# if backbone is endofm, the input needs to have time dimension first
		if self.backbone == 'endofm':
			# target shape: (batchsize, 3, 1, height, width)
			x = x.unsqueeze(2)
		if self.backbone == 'endovit':
			x = self.featureNet.forward_features(x)
			# only take the class token
			x = x[:,0,:]
		else:
			x = self.featureNet(x)
		x = self.out_layer(x)
		x = x.view(B,S,-1)

		return [x]

class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()
		
	def forward(self, x):
		return x

