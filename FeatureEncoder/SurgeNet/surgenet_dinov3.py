import torch
from transformers import AutoImageProcessor, AutoModel
import timm
 
# # Load model
# model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
 
# # Load weights
# weights_path = r"/projects/prjs0797/Yiping/SurgeNet_Phase/train_scripts/SurgeNet/hf_dinov3_vitb_123749_modified.pth"
# weights = torch.load(weights_path)
 
# # Change backbone.patch_embed wiith 'embeddings' and 'backbone.blocks' with 'layer'
# new_weights = {}
# for key, value in weights.items():
#     new_key = key  
#     if key.startswith('backbone.patch_embed'):
#         new_key = key.replace('backbone.patch_embed', 'embeddings')
#     elif key.startswith('backbone.blocks'):
#         new_key = key.replace('backbone.blocks', 'layer')
#     new_weights[new_key] = value
 
# # Load weights into model
# msg = model.load_state_dict(new_weights, strict=False)
# print(msg)

# # example input and output
# image_processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
# inputs = image_processor(images=torch.randn(1, 3, 256, 256),
#                             return_tensors="pt")
# outputs = model(**inputs)
# print(outputs.last_hidden_state.shape)  # should be (1, num_patches + 1, hidden_size)
# print(outputs.pooler_output.shape)  # should be (1, hidden_size)
class SurgeNet_DINOv3_vits_Feature(torch.nn.Module):
    def __init__(self, pretrained=True, **kwargs):
        super(SurgeNet_DINOv3_vits_Feature, self).__init__()
        if pretrained:
            weights_path = r"/projects/prjs1363/SurgPhaseBench/FeatureEncoder/SurgeNet/SurgeNet-Dinov3/dinov3_vits16_size336_surgenetxl_epoch15.pth"
            weights = torch.load(weights_path)
            self.model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
            new_weights = {}
            for key, value in weights.items():
                new_key = key  
                if key.startswith('backbone.patch_embed'):
                    new_key = key.replace('backbone.patch_embed', 'embeddings')
                elif key.startswith('backbone.blocks'):
                    new_key = key.replace('backbone.blocks', 'layer')
                new_weights[new_key] = value
            msg = self.model.load_state_dict(new_weights, strict=False)
            print(msg)
            print("Loaded pretrained DINOv3 model from Hugging Face with custom weights")
            self.model.cuda()
        else:
            raise NotImplementedError("Only pretrained model supported")
    
    def forward(self, x):
        # forward pass through the model
        outputs = self.model(x)
        pooled_features = outputs.pooler_output  # (batch_size, hidden_size)
        return pooled_features

class SurgeNet_DINOv3_vitb_Feature(torch.nn.Module):
    def __init__(self, pretrained=True, **kwargs):
        super(SurgeNet_DINOv3_vitb_Feature, self).__init__()
        if pretrained:
            weights_path = r"/projects/prjs1363/SurgPhaseBench/FeatureEncoder/SurgeNet/SurgeNet-Dinov3/dinov3_vitb16_size336_surgenetxl_epoch15.pth"
            weights = torch.load(weights_path)
            self.model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
            new_weights = {}
            for key, value in weights.items():
                new_key = key  
                if key.startswith('backbone.patch_embed'):
                    new_key = key.replace('backbone.patch_embed', 'embeddings')
                elif key.startswith('backbone.blocks'):
                    new_key = key.replace('backbone.blocks', 'layer')
                new_weights[new_key] = value
            msg = self.model.load_state_dict(new_weights, strict=False)
            print(msg)
            print("Loaded pretrained DINOv3 model from Hugging Face with custom weights")
            self.model.cuda()
        else:
            raise NotImplementedError("Only pretrained model supported")
    
    def forward(self, x):
        # forward pass through the model
        outputs = self.model(x)
        pooled_features = outputs.pooler_output  # (batch_size, hidden_size)
        return pooled_features

class SurgeNet_DINOv3_vitl_Feature(torch.nn.Module):
    def __init__(self, pretrained=True, **kwargs):
        super(SurgeNet_DINOv3_vitl_Feature, self).__init__()
        if pretrained:
            weights_path = r"/projects/prjs1363/SurgPhaseBench/FeatureEncoder/SurgeNet/SurgeNet-Dinov3/dinov3_vitl16_size336_surgenetxl_epoch15.pth"
            weights = torch.load(weights_path)
            self.model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
            new_weights = {}
            for key, value in weights.items():
                new_key = key  
                if key.startswith('backbone.patch_embed'):
                    new_key = key.replace('backbone.patch_embed', 'embeddings')
                elif key.startswith('backbone.blocks'):
                    new_key = key.replace('backbone.blocks', 'layer')
                new_weights[new_key] = value
            msg = self.model.load_state_dict(new_weights, strict=False)
            print(msg)
            print("Loaded pretrained DINOv3 model from Hugging Face with custom weights")
            self.model.cuda()
        else:
            raise NotImplementedError("Only pretrained model supported")
    
    def forward(self, x):
        # forward pass through the model
        outputs = self.model(x)
        pooled_features = outputs.pooler_output  # (batch_size, hidden_size)
        return pooled_features

class SurgeNet_DINOv1_vits_Feature(torch.nn.Module):
    '''
    DINOv1 Vision Transformer Small (ViT-S/16) Feature Extractor
    Uses timm to load the pretrained model weights.
    224x224 input size, patch size 16
    Outputs the CLS token feature vector, dim=384
    '''
    def __init__(self, pretrained=True, **kwargs):
        super(SurgeNet_DINOv1_vits_Feature, self).__init__()
        model_name = 'vit_small_patch16_224.dino'
        if pretrained:
            self.model = timm.create_model(
                model_name=model_name,
                img_size=224,
                patch_size=16,
                pretrained=True
            )
            self.model.cuda()
            print(f"Loaded pretrained DINOv1 model: {model_name} from timm")
            weights_path = "/projects/prjs1363/SurgPhaseBench/FeatureEncoder/SurgeNet/SurgeNet-Dinov3/dinov1_vits16_size224_surgenetxl_epoch50.pth"
            ckpt = torch.load(weights_path, weights_only=False)
            # if it's a full DINO checkpoint, pick the teacher or student dict
            if "teacher" in ckpt:
                state_dict = ckpt["teacher"]
            else:
                state_dict = ckpt

            # rename keys (remove 'backbone.' prefix)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("backbone."):
                    new_k = k[len("backbone."):]
                else:
                    new_k = k
                new_state_dict[new_k] = v
            # load into your model
            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            print("Loaded custom trained weights for DINOv1 ViT-B/16")
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
        else:
            raise NotImplementedError("Only pretrained model supported")
    
    def forward(self, x):
        # forward pass through the model
        features = self.model.forward_features(x)  # (batch_size, hidden_size)
        class_token_feature = features[:, 0, :]  # CLS token
        return class_token_feature

class SurgeNet_DINOv1_vitb_Feature(torch.nn.Module):
    '''
    DINOv1 Vision Transformer Small (ViT-S/16) Feature Extractor
    Uses timm to load the pretrained model weights.
    224x224 input size, patch size 16
    Outputs the CLS token feature vector, dim=384
    '''
    def __init__(self, pretrained=True, **kwargs):
        super(SurgeNet_DINOv1_vitb_Feature, self).__init__()
        model_name = 'vit_small_patch16_224.dino'
        model_name = 'vit_base_patch16_224.dino'
        if pretrained:
            self.model = timm.create_model(
                model_name=model_name,
                img_size=224,
                patch_size=16,
                pretrained=True
            )
            self.model.cuda()
            print(f"Loaded pretrained DINOv1 model: {model_name} from timm")
            weights_path = "/projects/prjs1363/SurgPhaseBench/FeatureEncoder/SurgeNet/SurgeNet-Dinov3/dinov1_vitb16_size224_surgenetxl_epoch50.pth"
            ckpt = torch.load(weights_path, weights_only=False)
            # if it's a full DINO checkpoint, pick the teacher or student dict
            if "teacher" in ckpt:
                state_dict = ckpt["teacher"]
            else:
                state_dict = ckpt

            # rename keys (remove 'backbone.' prefix)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("backbone."):
                    new_k = k[len("backbone."):]
                else:
                    new_k = k
                new_state_dict[new_k] = v
            # load into your model
            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            print("Loaded custom trained weights for DINOv1 ViT-B/16")
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        else:
            raise NotImplementedError("Only pretrained model supported")
    
    def forward(self, x):
        # forward pass through the model
        features = self.model.forward_features(x)  # (batch_size, hidden_size)
        class_token_feature = features[:, 0, :]  # CLS token
        return class_token_feature

class SurgeNet_DINOv2_vits_Feature(torch.nn.Module):
    '''
    DINOv2 Vision Transformer Small (ViT-S/14) Feature Extractor
    Uses timm to load the pretrained model weights.
    336x336 input size, patch size 14
    Outputs the CLS token feature vector, dim=384
    '''
    def __init__(self, pretrained=True, **kwargs):
        super(SurgeNet_DINOv2_vits_Feature, self).__init__()
        model_name = 'vit_small_patch14_dinov2'
        if pretrained:
            self.model = timm.create_model(
                model_name=model_name,
                img_size=336,
                patch_size=14,
                pretrained=True
            )
            print(f"Loaded pretrained DINOv1 model: {model_name} from timm")
            weights_path = "/projects/prjs1363/SurgPhaseBench/FeatureEncoder/SurgeNet/SurgeNet-Dinov3/dinov2_vits14_size336_surgenetxl_epoch50.pth"
            ckpt = torch.load(weights_path, weights_only=False)
            # if it's a full DINO checkpoint, pick the teacher or student dict
            if "teacher" in ckpt:
                state_dict = ckpt["teacher"]
            else:
                state_dict = ckpt

            # rename keys (remove 'backbone.' prefix)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("backbone."):
                    new_k = k[len("backbone."):]
                else:
                    new_k = k
                new_state_dict[new_k] = v
            # load into your model
            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            print("Loaded custom trained weights for SurgeNet DINOv2 ViT-S/14")
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
        else:
            raise NotImplementedError("Only pretrained model supported")
    
    def forward(self, x):
        # forward pass through the model
        features = self.model.forward_features(x)  # (batch_size, hidden_size)
        class_token_feature = features[:, 0, :]  # CLS token
        return class_token_feature

class SurgeNet_DINOv2_vitb_Feature(torch.nn.Module):
    '''
    DINOv2 Vision Transformer Base (ViT-B/14) Feature Extractor
    Uses timm to load the pretrained model weights.
    336x336 input size, patch size 14
    Outputs the CLS token feature vector, dim=768
    '''
    def __init__(self, pretrained=True, **kwargs):
        super(SurgeNet_DINOv2_vitb_Feature, self).__init__()
        model_name = 'vit_base_patch14_dinov2'
        if pretrained:
            self.model = timm.create_model(
                model_name=model_name,
                img_size=336,
                patch_size=14,
                pretrained=True
            )
            print(f"Loaded pretrained DINOv2 model: {model_name} from timm")
            weights_path = "/projects/prjs1363/SurgPhaseBench/FeatureEncoder/SurgeNet/SurgeNet-Dinov3/dinov2_vitb14_size336_surgenetxl_epoch50.pth"
            ckpt = torch.load(weights_path, weights_only=False)
            # if it's a full DINO checkpoint, pick the teacher or student dict
            if "teacher" in ckpt:
                state_dict = ckpt["teacher"]
            elif "backbone" in ckpt:
                state_dict = ckpt['backbone']
            else:
                state_dict = ckpt

            # rename keys (remove 'backbone.' prefix)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("backbone."):
                    new_k = k[len("backbone."):]
                else:
                    new_k = k
                new_state_dict[new_k] = v
            # load into your model
            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            print("Loaded custom trained weights for SurgeNet DINOv2 ViT-B/14")
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
        else:
            raise NotImplementedError("Only pretrained model supported")
    
    def forward(self, x):
        # forward pass through the model
        features = self.model.forward_features(x)  # (batch_size, hidden_size)
        class_token_feature = features[:, 0, :]  # CLS token
        return class_token_feature

class SurgeNet_DINOv2_vitl_Feature(torch.nn.Module):
    '''
    DINOv2 Vision Transformer Large (ViT-L/14) Feature Extractor
    Uses timm to load the pretrained model weights.
    336x336 input size, patch size 14
    Outputs the CLS token feature vector, dim=1024
    '''
    def __init__(self, pretrained=True, **kwargs):
        super(SurgeNet_DINOv2_vitl_Feature, self).__init__()
        model_name = 'vit_large_patch14_dinov2'
        if pretrained:
            self.model = timm.create_model(
                model_name=model_name,
                img_size=336,
                patch_size=14,
                pretrained=True
            )
            print(f"Loaded pretrained DINOv2 model: {model_name} from timm")
            weights_path = "/projects/prjs1363/SurgPhaseBench/FeatureEncoder/SurgeNet/SurgeNet-Dinov3/dinov2_vitl14_size336_surgenetxl_epoch30.pth"
            ckpt = torch.load(weights_path, weights_only=False)
            # if it's a full DINO checkpoint, pick the teacher or student dict
            if "teacher" in ckpt:
                state_dict = ckpt["teacher"]
            else:
                state_dict = ckpt

            # rename keys (remove 'backbone.' prefix)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("backbone."):
                    new_k = k[len("backbone."):]
                else:
                    new_k = k
                new_state_dict[new_k] = v
            # load into your model
            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            print("Loaded custom trained weights for SurgeNet DINOv2 ViT-L/14")
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
        else:
            raise NotImplementedError("Only pretrained model supported")
    
    def forward(self, x):
        # forward pass through the model
        features = self.model.forward_features(x)  # (batch_size, hidden_size)
        class_token_feature = features[:, 0, :]  # CLS token
        return class_token_feature

class DINOv3_vits_Feature(torch.nn.Module):
    '''
    DINOv3 Vision Transformer Small (ViT-S/16) Feature Extractor
    Uses transformers to load the pretrained model weights.
    336x336 input size, patch size 16
    Outputs the pooled feature vector, dim=384
    '''
    def __init__(self, pretrained=True, **kwargs):
        super(DINOv3_vits_Feature, self).__init__()
        if pretrained:
            self.model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
            self.model.cuda()
            print("Loaded pretrained DINOv3 model from Hugging Face")
        else:
            raise NotImplementedError("Only pretrained model supported")
    
    def forward(self, x):
        # forward pass through the model
        outputs = self.model(x)
        pooled_features = outputs.pooler_output  # (batch_size, hidden_size)
        return pooled_features

class DINOv3_vitb_Feature(torch.nn.Module):
    '''
    DINOv3 Vision Transformer Base (ViT-B/16) Feature Extractor
    Uses transformers to load the pretrained model weights.
    336x336 input size, patch size 16
    Outputs the pooled feature vector, dim=768
    '''

    def __init__(self, pretrained=True, **kwargs):
        super(DINOv3_vitb_Feature, self).__init__()
        if pretrained:
            self.model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
            self.model.cuda()
            print("Loaded pretrained DINOv3 model from Hugging Face")
        else:
            raise NotImplementedError("Only pretrained model supported")
    
    def forward(self, x):
        # forward pass through the model
        outputs = self.model(x)
        pooled_features = outputs.pooler_output  # (batch_size, hidden_size)
        return pooled_features

class DINOv3_vitl_Feature(torch.nn.Module):
    '''
    DINOv3 Vision Transformer Large (ViT-L/16) Feature Extractor
    Uses transformers to load the pretrained model weights.
    336x336 input size, patch size 16
    Outputs the pooled output feature vector, dim=1024
    '''
    def __init__(self, pretrained=True, **kwargs):
        super(DINOv3_vitl_Feature, self).__init__()
        if pretrained:
            self.model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
            self.model.cuda()
            print("Loaded pretrained DINOv3 model from Hugging Face")
        else:
            raise NotImplementedError("Only pretrained model supported")
    
    def forward(self, x):
        # forward pass through the model
        outputs = self.model(x)
        pooled_features = outputs.pooler_output  # (batch_size, hidden_size)
        return pooled_features
    
class DINOv1_vits_Feature(torch.nn.Module):
    '''
    DINOv1 Vision Transformer Small (ViT-S/16) Feature Extractor
    Uses timm to load the pretrained model weights.
    224x224 input size, patch size 16
    Outputs the CLS token feature vector, dim=384
    '''
    def __init__(self, pretrained=True, **kwargs):
        super(DINOv1_vits_Feature, self).__init__()
        model_name = 'vit_small_patch16_224.dino'
        if pretrained:
            self.model = timm.create_model(
                model_name=model_name,
                img_size=224,
                patch_size=16,
                pretrained=True
            )
            self.model.cuda()
            print(f"Loaded pretrained DINOv1 model: {model_name} from timm")
        else:
            raise NotImplementedError("Only pretrained model supported")
    
    def forward(self, x):
        # forward pass through the model
        features = self.model.forward_features(x)  # (batch_size, hidden_size)
        class_token_feature = features[:, 0, :]  # CLS token
        return class_token_feature

class DINOv1_vitb_Feature(torch.nn.Module):
    '''
    DINOv1 Vision Transformer Base (ViT-B/16) Feature Extractor
    Uses timm to load the pretrained model weights.
    224x224 input size, patch size 16
    Outputs the CLS token feature vector, dim=768
    '''
    def __init__(self, pretrained=True, **kwargs):
        super(DINOv1_vitb_Feature, self).__init__()
        model_name = 'vit_base_patch16_224.dino'
        if pretrained:
            self.model = timm.create_model(
                model_name=model_name,
                img_size=224,
                patch_size=16,
                pretrained=True
            )
            self.model.cuda()
            print(f"Loaded pretrained DINOv1 model: {model_name} from timm")
        else:
            raise NotImplementedError("Only pretrained model supported")
    
    def forward(self, x):
        # forward pass through the model
        features = self.model.forward_features(x)  # (batch_size, hidden_size)
        class_token_feature = features[:, 0, :]  # CLS token
        return class_token_feature


class DINOv2_vits_Feature(torch.nn.Module):
    '''
    DINOv2 Vision Transformer Small (ViT-S/14) Feature Extractor
    Uses timm to load the pretrained model weights.
    336x336 input size, patch size 14
    Outputs the CLS token feature vector, dim=384
    '''
    def __init__(self, pretrained=True, **kwargs):
        super(DINOv2_vits_Feature, self).__init__()
        model_name = 'vit_small_patch14_dinov2'
        if pretrained:
            self.model = timm.create_model(
                model_name=model_name,
                img_size=336,
                patch_size=14,
                pretrained=True
            )
            self.model.cuda()
            print(f"Loaded pretrained DINOv2 model: {model_name} from timm")
        else:
            raise NotImplementedError("Only pretrained model supported")
    
    def forward(self, x):
        # forward pass through the model
        features = self.model.forward_features(x)  # (batch_size, hidden_size)
        class_token_feature = features[:, 0, :]  # CLS token
        return class_token_feature

class DINOv2_vitb_Feature(torch.nn.Module):
    '''
    DINOv2 Vision Transformer Base (ViT-B/14) Feature Extractor
    Uses timm to load the pretrained model weights.
    336x336 input size, patch size 14
    Outputs the CLS token feature vector, dim=768
    '''
    def __init__(self, pretrained=True, **kwargs):
        super(DINOv2_vitb_Feature, self).__init__()
        model_name = 'vit_base_patch14_dinov2'
        if pretrained:
            self.model = timm.create_model(
                model_name=model_name,
                img_size=336,
                patch_size=14,
                pretrained=True
            )
            self.model.cuda()
            print(f"Loaded pretrained DINOv2 model: {model_name} from timm")
        else:
            raise NotImplementedError("Only pretrained model supported")
    
    def forward(self, x):
        # forward pass through the model
        features = self.model.forward_features(x)  # (batch_size, hidden_size)
        class_token_feature = features[:, 0, :]  # CLS token
        return class_token_feature


class DINOv2_vitl_Feature(torch.nn.Module):
    '''
    DINOv2 Vision Transformer Large (ViT-L/14) Feature Extractor
    Uses timm to load the pretrained model weights.
    336x336 input size, patch size 14
    Outputs the CLS token feature vector, dim=1024
    '''
    def __init__(self, pretrained=True, **kwargs):
        super(DINOv2_vitl_Feature, self).__init__()
        model_name = 'vit_large_patch14_dinov2'
        if pretrained:
            self.model = timm.create_model(
                model_name=model_name,
                img_size=336,
                patch_size=14,
                pretrained=True
            )
            self.model.cuda()
            print(f"Loaded pretrained DINOv2 model: {model_name} from timm")
        else:
            raise NotImplementedError("Only pretrained model supported")
    
    def forward(self, x):
        # forward pass through the model
        features = self.model.forward_features(x)  # (batch_size, hidden_size)
        class_token_feature = features[:, 0, :]  # CLS token
        return class_token_feature


class vanilla_vits_Feature(torch.nn.Module):
    '''
    DINOv2 Vision Transformer Small (ViT-S/14) Feature Extractor
    Uses timm to load the pretrained model weights.
    336x336 input size, patch size 14
    Outputs the CLS token feature vector, dim=384
    '''
    def __init__(self, pretrained=True, **kwargs):
        super(vanilla_vits_Feature, self).__init__()
        model_name = 'vit_small_patch14_dinov2'
        self.model = timm.create_model(
            model_name=model_name,
            img_size=336,
            patch_size=14,
            pretrained=True
        )
        
        print(f"Loaded pretrained DINOv2 model: {model_name} from timm")

        if not pretrained:
            self._reset_parameters()
        
        self.model.cuda()

    def _reset_parameters(self):
        """
        Reset model parameters (if supported by submodules).
        Also prints L2 difference to confirm reset.
        """
        before = {
            name: p.detach().clone()
            for name, p in self.model.named_parameters()
        }

        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        print("Reset model parameters to random initialization")

        for name, p in self.model.named_parameters():
            delta = torch.norm(p - before[name]).item()
            print(f"{name:40s} | change L2 = {delta:.6f}")
    
    def forward(self, x):
        # forward pass through the model
        features = self.model.forward_features(x)  # (batch_size, hidden_size)
        class_token_feature = features[:, 0, :]  # CLS token
        return class_token_feature


class vanilla_vitb_Feature(torch.nn.Module):
    '''
    DINOv2 Vision Transformer Small (ViT-S/14) Feature Extractor
    Uses timm to load the pretrained model weights.
    336x336 input size, patch size 14
    Outputs the CLS token feature vector, dim=384
    '''
    def __init__(self, pretrained=True, **kwargs):
        super(vanilla_vitb_Feature, self).__init__()
        model_name = 'vit_base_patch14_dinov2'
        self.model = timm.create_model(
            model_name=model_name,
            img_size=336,
            patch_size=14,
            pretrained=True
        )
        
        print(f"Loaded pretrained DINOv2 model: {model_name} from timm")

        if not pretrained:
            self._reset_parameters()
        
        self.model.cuda()
        
    def _reset_parameters(self):
        """
        Reset model parameters (if supported by submodules).
        Also prints L2 difference to confirm reset.
        """
        before = {
            name: p.detach().clone()
            for name, p in self.model.named_parameters()
        }

        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        print("Reset model parameters to random initialization")

        for name, p in self.model.named_parameters():
            delta = torch.norm(p - before[name]).item()
            print(f"{name:40s} | change L2 = {delta:.6f}")
    
    def forward(self, x):
        # forward pass through the model
        features = self.model.forward_features(x)  # (batch_size, hidden_size)
        class_token_feature = features[:, 0, :]  # CLS token
        return class_token_feature


class vanilla_vitl_Feature(torch.nn.Module):
    '''
    DINOv2 Vision Transformer Small (ViT-S/14) Feature Extractor
    Uses timm to load the pretrained model weights.
    336x336 input size, patch size 14
    Outputs the CLS token feature vector, dim=384
    '''
    def __init__(self, pretrained=True, **kwargs):
        super(vanilla_vitl_Feature, self).__init__()
        model_name = 'vit_large_patch14_dinov2'
        self.model = timm.create_model(
            model_name=model_name,
            img_size=336,
            patch_size=14,
            pretrained=True
        )
        
        print(f"Loaded pretrained DINOv2 model: {model_name} from timm")

        if not pretrained:
            self._reset_parameters()
        
        self.model.cuda()
        
    def _reset_parameters(self):
        """
        Reset model parameters (if supported by submodules).
        Also prints L2 difference to confirm reset.
        """
        before = {
            name: p.detach().clone()
            for name, p in self.model.named_parameters()
        }

        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        print("Reset model parameters to random initialization")

        for name, p in self.model.named_parameters():
            delta = torch.norm(p - before[name]).item()
            print(f"{name:40s} | change L2 = {delta:.6f}")
    
    def forward(self, x):
        # forward pass through the model
        features = self.model.forward_features(x)  # (batch_size, hidden_size)
        class_token_feature = features[:, 0, :]  # CLS token
        return class_token_feature
        
if __name__ == "__main__":
    # dummy input
    inputs = {'pixel_values': torch.randn(1, 3, 224, 224)}
    model = SurgeNet_DINOv3_vitl_Feature(pretrained=True)
    outputs = model(inputs['pixel_values'].cuda())
    print(outputs.shape)  # should be (1, hidden_size)