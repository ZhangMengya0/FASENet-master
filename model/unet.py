""" Full assembly of the parts to form the complete network """
"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=1)
    print(net)



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()
#
#         self.down1 = self.conv_stage(3, 8)
#         self.down2 = self.conv_stage(8, 16)
#         self.down3 = self.conv_stage(16, 32)
#         self.down4 = self.conv_stage(32, 64)
#         self.down5 = self.conv_stage(64, 128)
#         self.down6 = self.conv_stage(128, 256)
#         self.down7 = self.conv_stage(256, 512)
#
#         self.center = self.conv_stage(512, 1024)
#         # self.center_res = self.resblock(1024)
#
#         self.up7 = self.conv_stage(1024, 512)
#         self.up6 = self.conv_stage(512, 256)
#         self.up5 = self.conv_stage(256, 128)
#         self.up4 = self.conv_stage(128, 64)
#         self.up3 = self.conv_stage(64, 32)
#         self.up2 = self.conv_stage(32, 16)
#         self.up1 = self.conv_stage(16, 8)
#
#         self.trans7 = self.upsample(1024, 512)
#         self.trans6 = self.upsample(512, 256)
#         self.trans5 = self.upsample(256, 128)
#         self.trans4 = self.upsample(128, 64)
#         self.trans3 = self.upsample(64, 32)
#         self.trans2 = self.upsample(32, 16)
#         self.trans1 = self.upsample(16, 8)
#
#         self.conv_last = nn.Sequential(
#             nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()
#         )
#
#         self.max_pool = nn.MaxPool2d(2)
#
#     def conv_stage(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
#         if useBN:
#             return nn.Sequential(
#                 nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
#                 nn.BatchNorm2d(dim_out),
#                 nn.ReLU(),
#                 nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
#                 nn.BatchNorm2d(dim_out),
#                 nn.ReLU()
#             )
#         else:
#             return nn.Sequential(
#                 nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
#                 nn.ReLU(),
#                 nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
#                 nn.ReLU()
#             )
#
#     def upsample(self, ch_coarse, ch_fine):
#         return nn.Sequential(
#             nn.ConvTranspose2d(ch_coarse, ch_fine, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         conv1_out = self.down1(x)
#         conv2_out = self.down2(self.max_pool(conv1_out))
#         conv3_out = self.down3(self.max_pool(conv2_out))
#         conv4_out = self.down4(self.max_pool(conv3_out))
#         conv5_out = self.down5(self.max_pool(conv4_out))
#         conv6_out = self.down6(self.max_pool(conv5_out))
#         conv7_out = self.down7(self.max_pool(conv6_out))
#
#         out = self.center(self.max_pool(conv7_out))
#         # out = self.center_res(out)
#
#         out = self.up7(torch.cat([self.trans7(out), conv7_out], dim=1))
#         out = self.up6(torch.cat([self.trans6(out), conv6_out], dim=1))
#         out = self.up5(torch.cat([self.trans5(out), conv5_out], dim=1))
#         out = self.up4(torch.cat([self.trans4(out), conv4_out], dim=1))
#         out = self.up3(torch.cat([self.trans3(out), conv3_out], dim=1))
#         out = self.up2(torch.cat([self.trans2(out), conv2_out], dim=1))
#         out = self.up1(torch.cat([self.trans1(out), conv1_out], dim=1))
#
#         out = self.conv_last(out)
#
#         return out

