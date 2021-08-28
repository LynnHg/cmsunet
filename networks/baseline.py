import os
import torch
import math
from torch import nn
import torch.nn.functional as F
from networks.custom_modules.basic_modules import *


'''
================================================================
Total params: 59,393,538
Trainable params: 59,393,538
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 116.25
Params size (MB): 226.57
Estimated Total Size (MB): 342.81
----------------------------------------------------------------
'''


class Baseline(nn.Module):
    def __init__(self, img_ch=1, num_classes=6, depth=3):
        super(Baseline, self).__init__()
        chs = [44, 88, 176, 352, 704]

        self.pool = nn.MaxPool2d(2, 2)
        # p1 encoder
        self.p1_enc1 = EncoderBlock(img_ch, chs[0], depth=depth)
        self.p1_enc2 = EncoderBlock(chs[0], chs[1], depth=depth)
        self.p1_enc3 = EncoderBlock(chs[1], chs[2], depth=depth)
        self.p1_enc4 = EncoderBlock(chs[2], chs[3], depth=depth)
        self.p1_cen = EncoderBlock(chs[3], chs[4], depth=depth)

        self.dec4 = DecoderBlock(chs[4] * 3, chs[3])
        self.decconv4 = EncoderBlock(chs[3] * 4, chs[3])
        self.dec3 = DecoderBlock(chs[3], chs[2])
        self.decconv3 = EncoderBlock(chs[2] * 4, chs[2])

        self.dec2 = DecoderBlock(chs[2], chs[1])
        self.decconv2 = EncoderBlock(chs[1] * 4, chs[1])

        self.dec1 = DecoderBlock(chs[1], chs[0])
        self.decconv1 = EncoderBlock(chs[0] * 4, chs[0])

        self.conv_1x1 = nn.Conv2d(chs[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2, x3):
        # p1 encoder
        p1_x1 = self.p1_enc1(x1)
        p1_x2 = self.pool(p1_x1)
        p1_x2 = self.p1_enc2(p1_x2)
        p1_x3 = self.pool(p1_x2)
        p1_x3 = self.p1_enc3(p1_x3)
        p1_x4 = self.pool(p1_x3)
        p1_x4 = self.p1_enc4(p1_x4)
        p1_center = self.pool(p1_x4)
        p1_center = self.p1_cen(p1_center)

        # p2 encoder
        p2_x1 = self.p1_enc1(x2)
        p2_x2 = self.pool(p2_x1)
        p2_x2 = self.p1_enc2(p2_x2)
        p2_x3 = self.pool(p2_x2)
        p2_x3 = self.p1_enc3(p2_x3)
        p2_x4 = self.pool(p2_x3)
        p2_x4 = self.p1_enc4(p2_x4)
        p2_center = self.pool(p2_x4)
        p2_center = self.p1_cen(p2_center)

        # p3 encoder
        p3_x1 = self.p1_enc1(x3)
        p3_x2 = self.pool(p3_x1)
        p3_x2 = self.p1_enc2(p3_x2)
        p3_x3 = self.pool(p3_x2)
        p3_x3 = self.p1_enc3(p3_x3)
        p3_x4 = self.pool(p3_x3)
        p3_x4 = self.p1_enc4(p3_x4)
        p3_center = self.pool(p3_x4)
        p3_center = self.p1_cen(p3_center)

        fuse_center = torch.cat([p1_center, p2_center, p3_center], dim=1)
        fuse4 = torch.cat([p1_x4, p2_x4, p3_x4], dim=1)
        fuse3 = torch.cat([p1_x3, p2_x3, p3_x3], dim=1)
        fuse2 = torch.cat([p1_x2, p2_x2, p3_x2], dim=1)
        fuse1 = torch.cat([p1_x1, p2_x1, p3_x1], dim=1)

        d4 = self.dec4(fuse_center)
        d4 = torch.cat((fuse4, d4), dim=1)
        d4 = self.decconv4(d4)

        d3 = self.dec3(d4)
        d3 = torch.cat((fuse3, d3), dim=1)
        d3 = self.decconv3(d3)

        d2 = self.dec2(d3)
        d2 = torch.cat((fuse2, d2), dim=1)
        d2 = self.decconv2(d2)

        d1 = self.dec1(d2)
        d1 = torch.cat((fuse1, d1), dim=1)
        d1 = self.decconv1(d1)

        d1 = self.conv_1x1(d1)

        return d1

if __name__ == '__main__':
    from torchsummary import summary

    x1 = torch.randn([2, 1, 64, 64]).cuda()
    x2 = torch.randn([2, 1, 64, 64]).cuda()
    x3 = torch.randn([2, 1, 64, 64]).cuda()
    net = Baseline(num_classes=6).cuda()
    summary(net, input_size=[(1, 64, 64), (1, 64, 64), (1, 64, 64)])
    pred = net(x1, x2, x3)
    print(pred.shape)
