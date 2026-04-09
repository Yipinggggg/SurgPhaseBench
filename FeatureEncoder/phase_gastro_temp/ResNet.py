"""IMPORT PACKAGES"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math
import os
import argparse
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, opt, inference):
        super(Model, self).__init__()

        # Define Backbone architecture
        if opt.backbone == 'ResNet-50-ImageNet':
            url = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
            self.backbone = ResNet50(num_classes=opt.num_classes, channels=3, pretrained='ImageNet', url=url)
        elif opt.backbone == 'ResNet-50-GastroNet':
            self.backbone = ResNet50(num_classes=opt.num_classes, channels=3, pretrained='GastroNet', url=r'D:\OneDrive - TU Eindhoven\PhD\jaar 1\models\checkpoint_200ep_teacher_adapted.pth')
        elif opt.backbone == 'ResNet-101':
            url = "https://download.pytorch.org/models/resnet101-cd907fc2.pth"
            self.backbone = ResNet101(num_classes=opt.num_classes, channels=3, pretrained='ImageNet', url=url)
        elif opt.backbone == 'ResNet-152':
            url = "https://download.pytorch.org/models/resnet152-f82ba261.pth"
            self.backbone = ResNet152(num_classes=opt.num_classes, channels=3, pretrained=None)
        else:
            raise Exception('Unexpected Backbone {}'.format(opt.backbone))



    def forward(self, img):

        # Backbone output
        cls, low_level, high_level = self.backbone(img)

        return cls


""""""""""""""""""""""""""""""
"""" DEFINE BACKBONE MODELS"""
""""""""""""""""""""""""""""""
# https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py


# Class for creating BottleNeck Modules
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, track_running_stats=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride),
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, track_running_stats=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=(1, 1),
                               stride=(1, 1), padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion, track_running_stats=True)

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.relu(self.bn2(self.conv2(x)))

        x = self.conv3(x)
        x = self.bn3(x)

        # downsample if needed
        if self.downsample is not None:
            identity = self.downsample(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x


# Class for defining Block in ResNet
class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1,
                               stride=(stride, stride), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, track_running_stats=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1,
                               stride=(stride, stride), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, track_running_stats=True)

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)
        return x


# Class for Constructing complete ResNet Model
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3, pretrained=None, url=''):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=True)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

        # Initialize weights
        self._init_weight()

        # Define URL for pretrained weights
        self.url = url

        # Load pretrained weights if pretrained is True
        if pretrained:
            self._load_pretrained_model(pretrained)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        high_level_feat = x

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=(1, 1), stride=(stride, stride), bias=False),
                nn.BatchNorm2d(planes * ResBlock.expansion, track_running_stats=True)
            )

        layers.append(ResBlock(self.in_channels, planes, downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self, pretrained):

        # Define initialization
        if pretrained == 'ImageNet':
            pretrain_dict = model_zoo.load_url(self.url)
        elif pretrained == 'GastroNet':
            pretrain_dict = torch.load(self.url)

        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict and 'fc.weight' not in k and 'fc.bias' not in k:
                if 'running' not in k and 'batches' not in k and 'bn' not in k:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


# Functions to create various different versions of ResNet
def ResNet50(num_classes, channels=3, pretrained=None, url=''):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, channels, pretrained, url)


def ResNet101(num_classes, channels=3, pretrained=None, url=''):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, channels, pretrained, url)


def ResNet152(num_classes, channels=3, pretrained=None, url=''):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, channels, pretrained, url)


"""""" """""" """""" """""" """""" """"""
"""" DEFINE ResNet-Feature"""
"""""" """""" """""" """""" """""" """"""

class ResNetFeature(nn.Module):
    def __init__(self, ResBlock, layer_list, num_channels=3, pretrained=None, url=''):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=True)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

        # Initialize weights
        self._init_weight()

        # Define URL for pretrained weights
        self.url = url

        # Load pretrained weights if pretrained is True
        if pretrained:
            self._load_pretrained_model(pretrained)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        high_level_feat = x

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        # x = self.fc(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=(1, 1), stride=(stride, stride), bias=False),
                nn.BatchNorm2d(planes * ResBlock.expansion, track_running_stats=True)
            )

        layers.append(ResBlock(self.in_channels, planes, downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self, pretrained):

        # Define initialization
        if pretrained == 'ImageNet':
            pretrain_dict = model_zoo.load_url(self.url)
        elif pretrained == 'GastroNet':
            pretrain_dict = torch.load(self.url)

        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict and 'fc.weight' not in k and 'fc.bias' not in k:
                if 'running' not in k and 'batches' not in k and 'bn' not in k:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


# Functions to create various different versions of ResNet
def ResNet50Feature(channels=3, pretrained=None, url=''):
    return ResNetFeature(Bottleneck, [3, 4, 6, 3], channels, pretrained, url)

if __name__ == '__main__':

    # Specify nr. of classes and weights path
    num_classes = 4 
    weights_path = "/projects/0/prjs0797/Yiping/surgenet_phase/train_scripts/phase_gastro_temp/RN50_GastroNet-5M_DINOv1.pth"

    # Define model
    model = ResNet50Feature(channels=3, pretrained=None, url='').cuda()
    weights = torch.load(weights_path)
    msg = model.load_state_dict(weights, strict=False)
    print(msg)

    # Define input
    dummy = torch.zeros([4, 3, 256, 256]).cuda()
    out = model(dummy)
    print(out.shape)