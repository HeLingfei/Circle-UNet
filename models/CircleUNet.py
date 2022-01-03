from torch import nn
import torch
import torch.nn.functional as F
from models.UNet_utils import ConvBlock


class UPConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UPConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CircleUNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, circle_nums=(2, 3)):
        super(CircleUNet, self).__init__()
        self.circle_nums = circle_nums

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.max_up_pool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(ch_in=img_ch, ch_out=64)
        self.Conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.Conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.Conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.Conv5 = ConvBlock(ch_in=512, ch_out=1024)

        self.UpConv5_1 = UPConv(ch_in=1024, ch_out=512)
        self.UpCircle5 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.UPConv5_2 = UPConv(ch_in=1024, ch_out=512)

        self.UpConv4_1 = UPConv(ch_in=512, ch_out=256)
        self.UpCircle4 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.UPConv4_2 = UPConv(ch_in=512, ch_out=256)

        self.UpConv3_1 = UPConv(ch_in=256, ch_out=128)
        self.UpCircle3 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.UpConv3_2 = UPConv(ch_in=256, ch_out=128)

        self.UpConv2_1 = UPConv(ch_in=128, ch_out=64)
        self.UpCircle2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.UpConv2_2 = UPConv(ch_in=128, ch_out=64)

        self.UpConv1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        # origin_x = x.clone()
        x1 = self.Conv1(x)
        x2_1, id2 = self.max_pool(x1)

        x2 = self.Conv2(x2_1)
        x3_1, id3 = self.max_pool(x2)

        x3 = self.Conv3(x3_1)
        x4_1, id4 = self.max_pool(x3)

        x4 = self.Conv4(x4_1)
        x5_1, id5 = self.max_pool(x4)

        x5 = self.Conv5(x5_1)
        x6, id6 = self.max_pool(x5)

        # decoding + concat path
        d5 = self.max_up_pool(x6, indices=id6)
        d5 = self.UpConv5_1(d5)
        for i in range(self.circle_nums[-1]):
            d5 = torch.cat((x5_1, d5), dim=1)
            d5 = self.UpCircle5(d5)
        d5 = torch.cat((x5_1, d5), dim=1)
        d5 = self.UPConv5_2(d5)

        d4 = self.max_up_pool(d5, indices=id5)
        d4 = self.UpConv4_1(d4)
        for i in range(self.circle_nums[-2]):
            d4 = torch.cat((x4_1, d4), dim=1)
            d4 = self.UpCircle4(d4)
        d4 = torch.cat((x4_1, d4), dim=1)
        d4 = self.UPConv4_2(d4)

        d3 = self.max_up_pool(d4, indices=id4)
        d3 = self.UpConv3_1(d3)
        for i in range(self.circle_nums[-3]):
            d3 = torch.cat((x3_1, d3), dim=1)
            d3 = self.UpCircle3(d3)
        d3 = torch.cat((x3_1, d3), dim=1)
        d3 = self.UpConv3_2(d3)

        d2 = self.max_up_pool(d3, indices=id3)
        d2 = self.UpConv2_1(d2)
        for i in range(self.circle_nums[-4]):
            d2 = torch.cat((x2_1, d2), dim=1)
            d2 = self.UpCircle2(d2)
        d2 = torch.cat((x2_1, d2), dim=1)
        d2 = self.UpConv2_2(d2)

        d1 = self.max_up_pool(d2, indices=id2)
        d1 = self.UpConv1(d1)
        d1 = torch.sigmoid(d1)
        return d1
