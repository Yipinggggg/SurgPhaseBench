import torch
import torchvision
from pathlib import Path

class GroupNorm32(torch.nn.GroupNorm):
	def __init__(self, num_channels, num_groups=32, **kargs):
		super().__init__(num_groups, num_channels, **kargs)


def _load_groupnorm_checkpoint(model_name: str):
	base_dir = Path(__file__).resolve().parent
	candidates = [
		base_dir / f"ImageNet-{model_name}-GN.pth.tar",
		base_dir / f"ImageNet-{model_name}-GN.pth",
	]

	for ckpt_path in candidates:
		if ckpt_path.exists():
			ckpt = torch.load(str(ckpt_path), map_location="cpu")
			if isinstance(ckpt, dict) and "state_dict" in ckpt:
				state_dict = ckpt["state_dict"]
			else:
				state_dict = ckpt
			return {k.replace("module.", ""): v for k, v in state_dict.items()}

	raise FileNotFoundError(
		f"Could not find GroupNorm checkpoint for {model_name}. Tried: "
		+ ", ".join(str(p) for p in candidates)
	)

def resnet18_gn(pretrained=True,**kwargs): # own imagenet pretraining (not used for final paper)
	model = torchvision.models.resnet18(norm_layer=GroupNorm32)
	if pretrained:
		state_dict = _load_groupnorm_checkpoint("ResNet18")
		model.load_state_dict(state_dict)
	return model

def resnet34_gn(pretrained=True,**kwargs): # own imagenet pretraining (not used for final paper)
	model = torchvision.models.resnet34(norm_layer=GroupNorm32)
	if pretrained:
		state_dict = _load_groupnorm_checkpoint("ResNet34")
		model.load_state_dict(state_dict)
	return model

def resnet50_gn(pretrained=True,**kwargs):
	model = torchvision.models.resnet50(norm_layer=GroupNorm32)
	if pretrained:
		state_dict = _load_groupnorm_checkpoint("ResNet50")
		model.load_state_dict(state_dict)
	return model