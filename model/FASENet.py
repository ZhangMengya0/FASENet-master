import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from functools import partial

nonlinearity = partial(F.relu, inplace=False)


class DUpsampling(nn.Module):
    def __init__(self, inplanes, scale, num_class=2, pad=0):
        super(DUpsampling, self).__init__()

        self.conv_w = nn.Conv2d(inplanes, num_class * scale * scale, kernel_size=1, padding=pad, bias=False)

        self.scale = scale

    def forward(self, x):
        x = self.conv_w(x)
        N, C, H, W = x.size()
        # N, W, H, C
        x_permuted = x.permute(0, 3, 2, 1)
        # N, W, H*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, W, H * self.scale, int(C / (self.scale))))

        x_permuted = x_permuted.permute(0, 2, 1, 3)

        x_permuted = x_permuted.contiguous().view(
            (N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))
        # N, C/(scale**2), H*scale, W*scale
        x = x_permuted.permute(0, 3, 1, 2)

        return x


class MSGFA(nn.Module):
    def __init__(self, channel):
        super(MSGFA, self).__init__()

        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))

        out = x + dilate1_out + dilate2_out + dilate3_out
        return out


class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out


class FAA(nn.Module):
    def __init__(self, in_ch, out_ch, size=32, k=3, k_size=3):
        super(FAA, self).__init__()
        # self.conv = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv0 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(in_ch, in_ch, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()

        self.size = size
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1)
        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        self.relu1 = nn.ReLU()
        self.avg_pool_x = nn.AdaptiveAvgPool2d((size // 2, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, size // 2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(size // 2, size // 2, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(size // 2, size // 2, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv111 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        # input = self.conv(input)
        x = self.avg_pool(input)
        x1 = self.conv0(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)  # (1,64,1)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)  # (1,1,64)
        out1 = torch.sum(torch.matmul(x1, x2), dim=1).unsqueeze(-1).unsqueeze(-1)  # (1,64,1,1)
        # x1 = x1.transpose(-1, -2).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2), x1.transpose(-1, -2)), dim=1).unsqueeze(-1).unsqueeze(-1)
        out2 = self.sigmoid(out2)

        out = self.mix(out1, out2)
        out = self.conv0(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        g = input * out

        yL, yH = self.wt(g)

        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]

        # n, c, h, w = y_HH.size()
        y1 = self.avg_pool_x(y_HL)  # [1, 64, 64, 1]
        y1 = self.sigmoid(y1)

        y2 = self.avg_pool_y(y_LH)  # [1, 64, 1, 64]
        y2 = self.sigmoid(y2)

        y12 = torch.matmul(y1, y2)
        y12 = F.interpolate(y12, [self.size, self.size], mode="bilinear", align_corners=True)

        s = self.conv111(g * y12)

        return s


class StripConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernelsize=3, pad=1):
        super(StripConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Conv2d(in_channels, in_channels, (1, kernelsize), padding=(0, pad))
        self.conv2 = nn.Conv2d(in_channels, in_channels, (kernelsize, 1), padding=(pad, 0))
        self.conv3 = nn.Conv2d(in_channels, in_channels, (kernelsize, 1), padding=(pad, 0))
        self.conv4 = nn.Conv2d(in_channels, in_channels, (1, kernelsize), padding=(0, pad))

        self.cbr = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)

        x1 = self.conv1(x) + self.conv2(x)
        x2 = self.inv_h_transform(self.conv3(self.h_transform(x))) + self.inv_v_transform(
            self.conv4(self.v_transform(x)))

        x = self.cbr(x1 + x2)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        return x

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


class SegHead(nn.Module):
    def __init__(self, in_channels, out_channels, numclass=2):
        super(SegHead, self).__init__()
        self.cbr1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(out_channels, numclass, kernel_size=1)

    def forward(self, x):
        x = self.cbr1(x)

        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        x = self.final_conv(x)

        return x


class FASENet(nn.Module):
    def __init__(self, num_class=1):
        super(FASENet, self).__init__()

        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        self.cbr = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.msgfa = MSGFA(512)

        self.gfd3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            DUpsampling(512, 2, 128)
        )
        self.gfd2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            DUpsampling(128, 2, 32)
        )
        self.gfd1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DUpsampling(32, 2, 32)
        )

        self.sed3 = StripConv(256, 128, 3, 1)
        self.sed2 = StripConv(128, 64, 5, 2)
        self.sed1 = StripConv(64, 32, 7, 3)

        self.faa1 = FAA(64, 32, 128, 3)
        self.faa2 = FAA(128, 128, 64, 5)
        self.faa3 = FAA(256, 512, 32, 7)

        self.gamma1 = nn.Parameter(torch.randn(1, 32, 1, 1))
        self.gamma2 = nn.Parameter(torch.randn(1, 128, 1, 1))
        self.gamma3 = nn.Parameter(torch.randn(1, 512, 1, 1))
        self.sigmoid = nn.Sigmoid()

        self.seghead = SegHead(64, 64, 2)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        down_e2 = F.interpolate(e2, [e3.size()[2], e3.size()[3]], mode="bilinear", align_corners=True)
        down_e1 = F.interpolate(e1, [e3.size()[2], e3.size()[3]], mode="bilinear", align_corners=True)

        y = torch.cat((down_e2, down_e1), dim=1)
        x = e3
        y = self.cbr(y)  # 192-->256
        x = torch.cat((x, y), dim=1)  # channels 512 32 32
        x = self.msgfa(x)

        h1 = self.faa1(e1)
        h2 = self.faa2(e2)
        h3 = self.faa3(e3)

        g1 = self.sigmoid(self.gamma1)
        g2 = self.sigmoid(self.gamma2)
        g3 = self.sigmoid(self.gamma3)

        y = self.sed3(e3)
        x = self.gfd3(x * g3 + h3 * (1 - g3))

        y = self.sed2(y)
        x = self.gfd2(x * g2 + h2 * (1 - g2))

        y = self.sed1(y)
        x = self.gfd1(x * g1 + h1 * (1 - g1))

        x = torch.cat((x, y), dim=1)

        x = self.seghead(x)

        return x


if __name__ == "__main__":
    device = torch.device("cuda:0")
    input = torch.rand(1, 3, 512, 512)
    input = input.to(device)
    net = FASENet()
    net.to(device)
    output = net(input)
    print(output.size())
