import torch
from torch import nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(
                  in_channels=in_channels,
                  out_channels=in_channels,
                  kernel_size=kernel_size,
                  groups=in_channels,
                  stride=stride,
                  padding=padding,
                  bias=bias
        )
        self.pointwise_conv = nn.Conv2d(
                  in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=1,
                  bias=bias
        )

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        return out


class EntryFlowBlock(nn.Module):
    def __init__(self, in_channels, out_channels, return_no_skip=False):
        super(EntryFlowBlock, self).__init__()
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, 2)
        self.layer1 = nn.Sequential(
                  DepthwiseSeparableConv(in_channels, out_channels, 3, 1, 1, False),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
                  DepthwiseSeparableConv(out_channels, out_channels, 3, 1, 1, False),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
                  DepthwiseSeparableConv(out_channels, out_channels, 3, 2, 0, False),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(True)
        )
        self.return_no_skip = return_no_skip

    def forward(self, x):
        skip = x.clone()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out = x + self.skip_conv(skip)
        if self.return_no_skip:
            return x, out
        else:
            return out


class EntryFlow(nn.Module):
    def __init__(self):
        super(EntryFlow, self).__init__()
        self.layer1 = nn.Sequential(
                  nn.Conv2d(1, 32, 3, 2, bias=False),
                  nn.BatchNorm2d(32),
                  nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
                  nn.Conv2d(32, 64, 3, bias=False),
                  nn.BatchNorm2d(64),
                  nn.ReLU(True)
        )
        self.block1 = EntryFlowBlock(64, 128, True)
        self.block2 = EntryFlowBlock(128, 256)
        self.block3 = EntryFlowBlock(256, 728)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        skip, x = self.block1(x)
        x = self.block2(x)
        out = self.block3(x)
        return out, skip


class MiddleFlowBlock(nn.Module):
    def __init__(self):
        super(MiddleFlowBlock, self).__init__()
        self.module = nn.Sequential(
                  nn.Sequential(
                            DepthwiseSeparableConv(728, 728, 3, 1, 1, False),
                            nn.BatchNorm2d(728),
                            nn.ReLU(True)
                  ) for _ in range(3)
        )

    def forward(self, x):
        skip = x.clone()
        out = self.module(x) + skip
        return out

class MiddleFLow(nn.Module):
    def __init__(self):
        super(MiddleFLow, self).__init__()
        self.module = nn.Sequential(
                  MiddleFlowBlock()
                  for _ in range(18)
        )

    def forward(self, x):
        out = self.module(x)


class ExitFlow(nn.Module):
    def __int__(self):
        super(ExitFlow, self).__int__()
        self.skip_conv = nn.Conv2d(1024, 1024, 1, 2)
        self.layer1 = nn.Sequential(
                  DepthwiseSeparableConv(728, 1024, 3, 1, 1, False),
                  nn.BatchNorm2d(1024),
                  nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
                  DepthwiseSeparableConv(1024, 3, 1, 1, False),
                  nn.BatchNorm2d(1024),
                  nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
                  DepthwiseSeparableConv(1024, 1024, 3, 2, 0, False),
                  nn.BatchNorm2d(1024),
                  nn.ReLU(True)
        )
        self.layer4 = nn.Sequential(
                  DepthwiseSeparableConv(1024, 1536, 3, 1, 1, False),
                  nn.BatchNorm2d(1536),
                  nn.ReLU(True)
        )
        self.layer5 = nn.Sequential(
                  DepthwiseSeparableConv(1536, 1536, 3, 2, 1, False),
                  nn.BatchNorm2d(1536),
                  nn.ReLU(True)
        )
        self.layer6 = nn.Sequential(
                  DepthwiseSeparableConv(1536, 2048, 3, 1, 1, False),
                  nn.BatchNorm2d(2048),
                  nn.ReLU(True)
        )

    def forward(self, x):
        skip = x.clone()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x + self.skip_conv(skip)
        x = self.layer4(x)
        x = self.layer5(x)
        out = self.layer6(x)
        return out


class Xception65Module(nn.Module):
    def __init__(self):
        super(Xception65Module, self).__init__()
        self.entryflow = EntryFlow()
        self.middleflow = MiddleFLow()
        self.exitflow = ExitFlow()

    def forward(self, x):
        x, skip = self.entryflow(x)
        x = self.middleflow(x)
        out = self.exitflow(x)
        return out, skip


class ASPPModule(nn.Module):
    def __init__(self):
        super(ASPPModule, self).__init__()
        self.pointwise_conv = nn.Sequential(
                  nn.Conv2d(2048, 256, 1),
                  nn.BatchNorm2d(256),
                  nn.ReLU(True)
        )
        self.rate6_conv = nn.Sequential(
                  nn.Conv2d(2048, 256, 3, padding="same", dilation=6),
                  nn.BatchNorm2d(256),
                  nn.ReLU(True)
        )
        self.rate12_conv = nn.Sequential(
                  nn.Conv2d(2048, 256, 3, padding="same", dilation=12),
                  nn.BatchNorm2d(256),
                  nn.ReLU(True)
        )
        self.rate18_conv = nn.Sequential(
                  nn.Conv2d(2048, 256, 3, padding="same", dilation=18),
                  nn.BatchNorm2d(256),
                  nn.ReLU(True)
        )
        self.image_pooling = nn.Sequential(
                  nn.AdaptiveAvgPool2d(1),
                  nn.UpsamplingBilinear2d(16),
                  nn.Conv2d(2048, 256, 1, bias=False),
                  nn.BatchNorm2d(256),
                  nn.ReLU(True)
        )
        self.pointwise_after_aspp = nn.Sequential(
                  nn.Conv2d(4 * 256, 256, 1, bias=False)
        )

    def forward(self, x):
        pointwise = self.pointwise_conv(x)
        rate6 = self.rate6_conv(x)
        rate12 = self.rate12_conv(x)
        rate18 = self.rate18_conv(x)
        img_pool = self.image_pooling(x)
        out = torch.cat([pointwise, rate6, rate12, rate18, img_pool], dim=1)
        out = self.pointwise_after_aspp(out)
        return out


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.cat_conv = nn.Conv2d(2 * 256, num_classes, 3)
        self.skip_conv = nn.Conv2d(128, 256, 1)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, x, skip):
        skip = self.skip_conv(skip)
        x = self.upsampling(x)
        x = torch.cat([skip, x], dim=0)
        x = self.cat_conv(x)
        out = self.upsampling(x)
        return x


class DeepRetina(nn.Module):
    def __int__(self, num_classes):
        self.xception65 = Xception65Module()
        self.aspp = ASPPModule()
        self.decoder = Decoder(num_classes)

    def forward(self, x):
        x, skip = self.xception65(x)
        x = self.aspp(x)
        out = self.decoder(x, skip)