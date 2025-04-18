import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
from torchgeo.models import resnet18
from torchgeo.models import ResNet18_Weights


class DoubleConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsampling=True):
        super().__init__()
        if upsampling:
            self.upconv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        else:
            self.upconv = nn.Identity()  # does nothing
        self.layers = DoubleConv(in_channels, out_channels, out_channels)

    # def forward(self, x, skip_connection):


class UNet(nn.Module):
    def __init__(self, base_model, num_classes, channels=(13, 64, 128, 256, 512)):
        super().__init__()
        self.num_classes = num_classes
        self.base_model = base_model
        # encloder blocks
        self.encoder0 = nn.Sequential(
            self.base_model.conv1, self.base_model.bn1, self.base_model.relu
        )
        self.pool = self.base_model.maxpool
        self.encoder1 = self.base_model.layer1
        self.encoder2 = self.base_model.layer2
        self.encoder3 = self.base_model.layer3
        self.encoder4 = self.base_model.layer4

        # freezing the first 2 encoder blocks
        for param in self.encoder1.parameters():
            param.requires_grad = False
        for param in self.encoder2.parameters():
            param.requires_grad = False

        # decoder blocks
        # self.decoder_1 =
