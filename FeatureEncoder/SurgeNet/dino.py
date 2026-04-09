from .surgenet_dinov3 import *

def DINOv3_vits(pretrained=True,**kwargs): 
    if pretrained:
        model = DINOv3_vits_Feature(pretrained=True)
    else:
        raise NotImplementedError("Only pretrained model supported")
    
    return model

def DINOv3_vitl(pretrained=True,**kwargs):
    if pretrained:
        model = DINOv3_vitl_Feature(pretrained=True)
    else:
        raise NotImplementedError("Only pretrained model supported")
    
    return model

def DINOv3_vitb(pretrained=True,**kwargs):
    if pretrained:
        model = DINOv3_vitb_Feature(pretrained=True)
    else:
        raise NotImplementedError("Only pretrained model supported")
    
    return model

def DINOv2_vitl(pretrained=True,**kwargs): 
    if pretrained:
        model = DINOv2_vitl_Feature(pretrained=True)
        model
    else:
        raise NotImplementedError("Only pretrained model supported")
    return model

def DINOv2_vitb(pretrained=True,**kwargs): 
    if pretrained:
        model = DINOv2_vitb_Feature(pretrained=True)
        model
    else:
        raise NotImplementedError("Only pretrained model supported")
    return model

def DINOv2_vits(pretrained=True,**kwargs): 
    if pretrained:
        model = DINOv2_vits_Feature(pretrained=True)
        model
    else:
        raise NotImplementedError("Only pretrained model supported")
    return model

def DINOv1_vitb(pretrained=True,**kwargs): 
    if pretrained:
        model = DINOv1_vitb_Feature(pretrained=True)
        model
    else:
        raise NotImplementedError("Only pretrained model supported")
    return model

def DINOv1_vits(pretrained=True,**kwargs): 
    if pretrained:
        model = DINOv1_vits_Feature(pretrained=True)
        model
    else:
        raise NotImplementedError("Only pretrained model supported")
    return model

def SurgeNet_DINOv3_vits(pretrained=True,**kwargs):
    if pretrained:
        model = SurgeNet_DINOv3_vits_Feature(pretrained=True)
    else:
        raise NotImplementedError("Only pretrained model supported")
    
    return model

def SurgeNet_DINOv3_vitl(pretrained=True,**kwargs):
    if pretrained:
        model = SurgeNet_DINOv3_vitl_Feature(pretrained=True)
    else:
        raise NotImplementedError("Only pretrained model supported")
    
    return model
    
def SurgeNet_DINOv3_vitb(pretrained=True,**kwargs): 
    if pretrained:
        model = SurgeNet_DINOv3_vitb_Feature(pretrained=True)
    else:
        raise NotImplementedError("Only pretrained model supported")
    
    return model

def SurgeNet_DINOv2_vitl(pretrained=True,**kwargs):
    if pretrained:
        model = SurgeNet_DINOv2_vitl_Feature(pretrained=True)
    else:
        raise NotImplementedError("Only pretrained model supported")
    
    return model

def SurgeNet_DINOv2_vits(pretrained=True,**kwargs):
    if pretrained:
        model = SurgeNet_DINOv2_vits_Feature(pretrained=True)
    else:
        raise NotImplementedError("Only pretrained model supported")
    
    return model

def SurgeNet_DINOv2_vitb(pretrained=True,**kwargs):
    if pretrained:
        model = SurgeNet_DINOv2_vitb_Feature(pretrained=True)
    else:
        raise NotImplementedError("Only pretrained model supported")
    
    return model

def SurgeNet_DINOv1_vitb(pretrained=True,**kwargs):
    if pretrained:
        model = SurgeNet_DINOv1_vitb_Feature(pretrained=True)
    else:
        raise NotImplementedError("Only pretrained model supported")
    
    return model

def SurgeNet_DINOv1_vits(pretrained=True,**kwargs): 
    if pretrained:
        model = SurgeNet_DINOv1_vits_Feature(pretrained=True)
    else:
        raise NotImplementedError("Only pretrained model supported")
    
    return model

def vanilla_vits(pretrained=False,**kwargs):
    if not pretrained:
        model = vanilla_vits_Feature(pretrained=False)
    else:
        raise NotImplementedError("Only non-pretrained model supported")
    return model

def vanilla_vitl(pretrained=False,**kwargs):
    if not pretrained:
        model = vanilla_vitl_Feature(pretrained=False)
    else:
        raise NotImplementedError("Only non-pretrained model supported")
    return model

def vanilla_vitb(pretrained=False,**kwargs):
    if not pretrained:
        model = vanilla_vitb_Feature(pretrained=False)
    else:
        raise NotImplementedError("Only non-pretrained model supported")
    return model