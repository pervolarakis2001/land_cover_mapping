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
    def __init__(self, in_channels, skip_channels, out_channels, upsampling=True):
        super().__init__()
        self.upsampling = upsampling

        if upsampling:
            self.upconv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        else:
            self.upconv = nn.Identity()

        self.conv = DoubleConv(
            in_channels=out_channels + skip_channels,
            mid_channels=out_channels,
            out_channels=out_channels,
        )

    def forward(self, x, skip_connection):
        x = self.upconv(x)

        skip_interp = F.interpolate(
            skip_connection, size=x.shape[2:], mode="bilinear", align_corners=False
        )

        x = torch.cat([skip_interp, x], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, num_classes, channels=(13, 64, 128, 256, 512)):
        super().__init__()
        self.num_classes = num_classes
        self.resnet = resnet18(ResNet18_Weights.SENTINEL2_ALL_MOCO, pretrained=True)
        # encloder blocks
        self.encoder0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            nn.ReLU(inplace=True),
            self.resnet.maxpool,
        )
        self.encoder1 = self.resnet.layer1
        self.encoder2 = self.resnet.layer2
        self.encoder3 = self.resnet.layer3
        self.encoder4 = self.resnet.layer4
        del self.resnet

        self.bottleneck = DoubleConv(channels[4], channels[4], channels[4])

        self.decoder_1 = DecoderBlock(channels[4], channels[4], channels[3])
        self.decoder_2 = DecoderBlock(channels[3], channels[3], channels[2])
        self.decoder_3 = DecoderBlock(channels[2], channels[2], channels[1])
        self.decoder_4 = DecoderBlock(channels[1], channels[1], channels[1])

        # final layer
        self.final = nn.Sequential(
            nn.Conv2d(channels[1], num_classes, kernel_size=1),
        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder0(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        botx = self.bottleneck(x5)
        # Decoder
        up1 = self.decoder_1(botx, x5)
        up2 = self.decoder_2(up1, x4)
        up3 = self.decoder_3(up2, x3)
        up4 = self.decoder_4(up3, x2)

        # Final output
        outpt_mask = self.final(up4)
        outpt_mask = F.interpolate(
            outpt_mask, size=(256, 256), mode="bilinear", align_corners=False
        )

        return outpt_mask
