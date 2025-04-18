# model.py

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn as nn

def pretrainedModel(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

class ResidualBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv2 = nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_ch)
        
    def forward(self, x):
        residual = x
        x = nn.ReLU(inplace=True)(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return nn.ReLU(inplace=True)(x + residual)
    
def Model1(num_classes):
    backbone = nn.Sequential(
        nn.Conv2d(3, 64, 3, 2, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, 3, 2, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        ResidualBlock(128),
        nn.Conv2d(128, 256, 3, 1, 1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True)
    )
    backbone.out_channels = 256

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(['0'], 7, 2)

    return FasterRCNN(backbone, num_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler,
                    min_size=800, max_size=1333)

def Model2(num_classes):
    backbone = nn.Sequential(
        nn.Conv2d(3, 64, 3, 2, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        ResidualBlock(64),
        nn.Conv2d(64, 128, 3, 2, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        ResidualBlock(128),
        nn.Conv2d(128, 256, 3, 2, 1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Dropout2d(0.2),
        ResidualBlock(256)
    )
    backbone.out_channels = 256

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(['0'], 7, 2)
    
    return FasterRCNN(backbone, num_classes,
                     rpn_anchor_generator=anchor_generator,
                     box_roi_pool=roi_pooler)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Integrate SEBlock into the backbone
class ResNetWithSE(nn.Module):
    def __init__(self, backbone):
        super(ResNetWithSE, self).__init__()
        self.body = backbone.body
        self.fpn = backbone.fpn

        # Add SE blocks after Conv2d layers in backbone (optional)
        for name, module in self.body.named_children():
            if isinstance(module, nn.Sequential):
                for idx, layer in enumerate(module):
                    if isinstance(layer, nn.Conv2d):
                        module[idx] = nn.Sequential(
                            layer,
                            SEBlock(layer.out_channels)
                        )

        # Tell FasterRCNN how many channels your FPN returns
        self.out_channels = 256

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x

def Model6(num_classes):
    # Load a pre-trained ResNet50 backbone with FPN
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    # Wrap the backbone with SE blocks
    backbone_with_se = ResNetWithSE(backbone)
    
    # Define the anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),  # ← one per feature map
        aspect_ratios=((0.5, 1.0, 2.0),) * 5          # ← repeat aspect ratios 5 times
    )
            
    # Define the ROI pooler
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3', 'pool'],
        output_size=7,
        sampling_ratio=2
    )
    
    # Create the Faster R-CNN model
    model = FasterRCNN(
        backbone=backbone_with_se,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    
    return model
