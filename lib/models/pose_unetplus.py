# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False
    )

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DecoderBlock(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(DecoderBlock, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output

class PoseUNet(nn.Module):

    def __init__(self, block, layers, cfg, encoder_channels, **kwargs):
        self.inplanes = 64

        super(PoseUNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.UP = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        in_channels = encoder_channels[::-1]

        self.x01 = DecoderBlock(in_channels[0] + in_channels[1], in_channels[0], in_channels[0])
        self.x11 = DecoderBlock(in_channels[1] + in_channels[2], in_channels[1], in_channels[1])
        self.x21 = DecoderBlock(in_channels[2] + in_channels[3], in_channels[2], in_channels[2])

        self.x02 = DecoderBlock(in_channels[0] * 2 + in_channels[1], in_channels[0], in_channels[0])
        self.x12 = DecoderBlock(in_channels[1] * 2 + in_channels[2], in_channels[1], in_channels[1])

        self.x03 = DecoderBlock(in_channels[0] * 3 + in_channels[1], in_channels[0], in_channels[0])

        self.x31 = DecoderBlock(in_channels[3] + in_channels[4], in_channels[3], in_channels[3])
        self.x22 = DecoderBlock(in_channels[2] * 2 + in_channels[3], in_channels[2], in_channels[2])
        self.x13 = DecoderBlock(in_channels[1] * 3 + in_channels[2], in_channels[1], in_channels[1])
        self.x04 = DecoderBlock(in_channels[0] * 4 + in_channels[1], in_channels[0], in_channels[0])

        self.final_conv = nn.Conv2d(in_channels[0], 15, kernel_size=1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x00 = self.relu(x)

        x10 = self.maxpool(x00)
        x10 = self.layer1(x10)

        x20 = self.layer2(x10)
        x30 = self.layer3(x20)
        x40 = self.layer4(x30)

        x01 = self.x01(torch.cat([x00, self.UP(x10)], 1))

        x11 = self.x11(torch.cat([x10, self.UP(x20)], 1))
        x02 = self.x02(torch.cat([x00, x01, self.UP(x11)], 1))

        x21 = self.x21(torch.cat([x20, self.UP(x30)], 1))
        x12 = self.x12(torch.cat([x10, x11, self.UP(x21)], 1))
        x03 = self.x03(torch.cat([x00, x01, x02, self.UP(x12)], 1))

        x31 = self.x31(torch.cat([x30, self.UP(x40)], 1))
        x22 = self.x22(torch.cat([x20, x21, self.UP(x31)], 1))
        x13 = self.x13(torch.cat([x10, x11, x12, self.UP(x22)], 1))
        x04 = self.x04(torch.cat([x00, x01, x02, x03, self.UP(x13)], 1))

        output = self.final_conv(x04)
        return output

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            logger.info('=> init final conv weights from normal distribution')
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)

resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}

def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]

    model = PoseUNet(block_class, layers, cfg, encoder_channels=(2048, 1024, 512, 256, 64), **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model


if __name__ == '__main__':

    num_layers = 152

    block_class, layers = resnet_spec[num_layers]

    model = PoseUNet(block_class, layers, cfg=None, encoder_channels=(2048, 1024, 512, 256, 64))

    img = torch.randn(2,3,384,288)
    out = model(img)
    print(out.size())