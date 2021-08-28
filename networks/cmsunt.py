import os
import torch
import math
from torch import nn
import torch.nn.functional as F
from utils.misc import initialize_weights


class SplitConv(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, use_dilation=False):
        super(SplitConv, self).__init__()
        pad = (kernel_size-1) // 2
        dilation = 1
        if use_dilation:
            dilation = pad
        self.conv = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, (1, kernel_size), padding=(0, pad), dilation=(1, dilation)),
            nn.Conv2d(out_chs, out_chs, (kernel_size, 1), padding=(pad, 0), dilation=(dilation, 1))
        )

    def forward(self, x):
        return self.conv(x)


class MSCM(nn.Module):
    def __init__(self, in_chs, exp_chs, out_chs):
        super(MSCM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, 1, padding=0),
            nn.BatchNorm2d(out_chs),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            SplitConv(in_chs, exp_chs, 3),
            nn.BatchNorm2d(exp_chs),
            nn.ReLU(),
            nn.Conv2d(exp_chs, out_chs, 1, padding=0),
            nn.BatchNorm2d(out_chs),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            SplitConv(in_chs, exp_chs, 5),
            nn.BatchNorm2d(exp_chs),
            nn.ReLU(),
            nn.Conv2d(exp_chs, out_chs, 1, padding=0),
            nn.BatchNorm2d(out_chs),
            nn.ReLU()
        )

        self.bottleneck = nn.Conv2d(out_chs * 3, out_chs, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        x3 = self.conv5(x)

        out = torch.cat([x1, x2, x3], dim=1)

        out = self.bottleneck(out)

        return out


def divisor(a, b):
    if b == 0:
        return a
    else:
        return divisor(b, a % b)


class ChannelReconstructionUpsampling(nn.Module):
    def __init__(self, upscale_factor, ch_in, ch_out, ratio=2):
        super(ChannelReconstructionUpsampling, self).__init__()
        upsample_dim = (upscale_factor ** 2) * ch_out
        init_channels = math.ceil(upsample_dim / ratio)
        new_channels = init_channels * (ratio - 1)
        group = divisor(ch_in, new_channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, init_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, padding=1, groups=group, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True),
        )

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.pixel_shuffle(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=2, use_res=False):
        super(EncoderBlock, self).__init__()

        self.use_res = use_res

        self.conv = [nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )]

        for i in range(1, depth):
            self.conv.append(nn.Sequential(nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=False),
                                           nn.BatchNorm2d(ch_out),
                                           nn.Sequential() if use_res and i == depth-1 else nn.ReLU(inplace=True)
                                           ))
        self.conv = nn.Sequential(*self.conv)
        if use_res:
            self.conv1x1 = nn.Conv2d(ch_in, ch_out, 1)

    def forward(self, x):
        if self.use_res:
            residual = self.conv1x1(x)

        x = self.conv(x)

        if self.use_res:
            x += residual
            x = F.relu(x)

        return x


class CMSUNet(nn.Module):
    def __init__(self, img_ch=1, num_classes=6, depth=3):
        super(CMSUNet, self).__init__()
        chs = [44, 88, 176, 352, 704]

        self.pool = nn.MaxPool2d(2, 2)
        self.p1_msc = MSCM(img_ch, chs[1], chs[0])
        # p1 encoder
        self.p1_enc1 = EncoderBlock(chs[0], chs[0], depth=depth)
        self.p1_enc2 = EncoderBlock(chs[0], chs[1], depth=depth)
        self.p1_enc3 = EncoderBlock(chs[1], chs[2], depth=depth)
        self.p1_enc4 = EncoderBlock(chs[2], chs[3], depth=depth)
        self.p1_cen = EncoderBlock(chs[3], chs[4], depth=depth)

        self.dec4 = ChannelReconstructionUpsampling(2, chs[4] * 3, chs[3])
        self.decconv4 = EncoderBlock(chs[3] * 4, chs[3])

        self.dec3 = ChannelReconstructionUpsampling(2, chs[3], chs[2])
        self.decconv3 = EncoderBlock(chs[2] * 4, chs[2])

        self.dec2 = ChannelReconstructionUpsampling(2, chs[2], chs[1])
        self.decconv2 = EncoderBlock(chs[1] * 4, chs[1])

        self.dec1 = ChannelReconstructionUpsampling(2, chs[1], chs[0])
        self.decconv1 = EncoderBlock(chs[0] * 4, chs[0])

        self.conv_1x1 = nn.Conv2d(chs[0], num_classes, 1, bias=False)
        initialize_weights(self)

        if self.training:
            self.p1_duc = ChannelReconstructionUpsampling(16, chs[4], num_classes)
            self.p2_duc = ChannelReconstructionUpsampling(16, chs[4], num_classes)
            self.p3_duc = ChannelReconstructionUpsampling(16, chs[4], num_classes)

    def forward(self, x1, x2, x3):
        # p1 encoder
        p1_x1 = self.p1_msc(x1)
        p1_x1 = self.p1_enc1(p1_x1)
        p1_x2 = self.pool(p1_x1)
        p1_x2 = self.p1_enc2(p1_x2)
        p1_x3 = self.pool(p1_x2)
        p1_x3 = self.p1_enc3(p1_x3)
        p1_x4 = self.pool(p1_x3)
        p1_x4 = self.p1_enc4(p1_x4)
        p1_center = self.pool(p1_x4)
        p1_center = self.p1_cen(p1_center)

        # p2 encoder
        p2_x1 = self.p1_msc(x2)
        p2_x1 = self.p1_enc1(p2_x1)
        p2_x2 = self.pool(p2_x1)
        p2_x2 = self.p1_enc2(p2_x2)
        p2_x3 = self.pool(p2_x2)
        p2_x3 = self.p1_enc3(p2_x3)
        p2_x4 = self.pool(p2_x3)
        p2_x4 = self.p1_enc4(p2_x4)
        p2_center = self.pool(p2_x4)
        p2_center = self.p1_cen(p2_center)

        # p3 encoder
        p3_x1 = self.p1_msc(x3)
        p3_x1 = self.p1_enc1(p3_x1)
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

        if self.training:
            p1_aux = self.p1_duc(p1_center)
            p2_aux = self.p2_duc(p2_center)
            p3_aux = self.p3_duc(p3_center)

            return d1, (torch.sigmoid(p1_aux), torch.sigmoid(p2_aux), torch.sigmoid(p3_aux))
        else:
            return d1

if __name__ == '__main__':
    from torchsummary import summary

    x1 = torch.randn([2, 1, 64, 64]).cuda()
    x2 = torch.randn([2, 1, 64, 64]).cuda()
    x3 = torch.randn([2, 1, 64, 64]).cuda()
    net = CMSUNet(num_classes=6).cuda()
    net.train()
    summary(net, input_size=[(1, 64, 64), (1, 64, 64), (1, 64, 64)])
    pred, aux = net(x1, x2, x3)
    print(pred.shape)
