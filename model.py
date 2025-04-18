# model.py

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def pretrainedModel(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def Model1(num_classes):
    # Custom CNN Backbone
    backbone = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        nn.ReLU()
    )
    backbone.out_channels = 256

    # Anchor generator
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # ROI Pooler
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model

def Model2(num_classes):
    backbone = nn.Sequential(
        nn.Conv2d(3, 64, 3, 2, 1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, 2, 1),
        nn.ReLU(),
        nn.Conv2d(128, 256, 3, 2, 1),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Conv2d(256, 512, 3, 2, 1),
        nn.ReLU()
    )
    backbone.out_channels = 512

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(['0'], 7, 2)

    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model

class DepthwiseConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

def Model3(num_classes):
    backbone = nn.Sequential(
        DepthwiseConv(3, 32),
        nn.ReLU(),
        DepthwiseConv(32, 64),
        nn.ReLU(),
        DepthwiseConv(64, 128),
        nn.ReLU()
    )
    backbone.out_channels = 128

    anchor_generator = AnchorGenerator(sizes=((32, 64),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(['0'], 7, 2)

    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

def Model4(num_classes):
    backbone = nn.Sequential(
        nn.Conv2d(3, 64, 3, 2, 1),
        nn.ReLU(),
        nn.Dropout2d(0.25),
        nn.Conv2d(64, 128, 3, 2, 1),
        nn.ReLU(),
        nn.Dropout2d(0.3),
        nn.Conv2d(128, 256, 3, 2, 1),
        nn.ReLU(),
        nn.Dropout2d(0.4)
    )
    backbone.out_channels = 256

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(['0'], 7, 2)

    model = FasterRCNN(backbone, num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model

def Model5(num_classes):
    backbone = nn.Sequential(
        nn.Conv2d(3, 64, 3, 2, 1),
        nn.GroupNorm(8, 64),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, 2, 1),
        nn.GroupNorm(8, 128),
        nn.ReLU(),
        nn.Conv2d(128, 256, 3, 2, 1),
        nn.GroupNorm(8, 256),
        nn.ReLU()
    )
    backbone.out_channels = 256

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(['0'], 7, 2)

    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model


class BasicResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

def make_layer(in_ch, out_ch, blocks, stride=1):
    downsample = None
    if stride != 1 or in_ch != out_ch:
        downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_ch)
        )

    layers = [BasicResBlock(in_ch, out_ch, stride, downsample)]
    for _ in range(1, blocks):
        layers.append(BasicResBlock(out_ch, out_ch))
    return nn.Sequential(*layers)

def Model6(num_classes):
    backbone = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        make_layer(64, 64, 3),
        make_layer(64, 128, 4, stride=2),
        make_layer(128, 256, 6, stride=2),
        make_layer(256, 512, 3, stride=2),
    )
    backbone.out_channels = 512

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(['0'], 7, 2)

    model = FasterRCNN(backbone, num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model