from torch import nn


class CR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=None, bias=True):
        super(CR, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class CGR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=None, bias=True):
        super(CGR, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.gn = nn.GroupNorm(32, out_channel)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        return x


class CBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=None, bias=True):
        super(CBR, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class FPNExtractor(nn.Module):
    def __init__(self, c3, c4, c5, inner_channel=256, bias=True):
        super(FPNExtractor, self).__init__()
        self.c3_latent = nn.Conv2d(c3, inner_channel, 1, 1, 0, bias=bias)
        self.c4_latent = nn.Conv2d(c4, inner_channel, 1, 1, 0, bias=bias)
        self.c5_latent = nn.Conv2d(c5, inner_channel, 1, 1, 0, bias=bias)
        self.c5_to_c6 = nn.Conv2d(c5, inner_channel, 3, 2, 1, bias=bias)
        self.c6_to_c7 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(inner_channel, inner_channel, 3, 2, 1, bias=bias)
        )

    def forward(self, xs):
        c3, c4, c5 = xs
        f3 = self.c3_latent(c3)
        f4 = self.c4_latent(c4)
        f5 = self.c5_latent(c5)
        f6 = self.c5_to_c6(c5)
        f7 = self.c6_to_c7(f6)
        return [f3, f4, f5, f6, f7]


class FPN(nn.Module):
    def __init__(self, c3, c4, c5, out_channel, bias=True):
        super(FPN, self).__init__()
        self.fpn_extractor = FPNExtractor(c3, c4, c5, out_channel, bias)
        self.p3_out = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=bias)
        self.p4_out = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=bias)
        self.p5_out = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=bias)

    def forward(self, xs):
        f3, f4, f5, f6, f7 = self.fpn_extractor(xs)
        p5 = self.p5_out(f5)
        f4 = f4 + nn.UpsamplingBilinear2d(size=(f4.shape[2:]))(f5)
        p4 = self.p4_out(f4)
        f3 = f3 + nn.UpsamplingBilinear2d(size=(f3.shape[2:]))(f4)
        p3 = self.p3_out(f3)
        return [p3, p4, p5, f6, f7]
# class FPN(nn.Module):
#     def __init__(self, c3, c4, c5, out_channel=256):
#         super(FPN, self).__init__()
#
#         self.bones = nn.Sequential(
#             nn.Conv2d(c5, out_channel, 1, 1, 0),  # 0
#             nn.Conv2d(out_channel, out_channel, 3, 1, 1),  # 1
#
#             nn.Conv2d(c4, out_channel, 1, 1, 0),  # 2
#             nn.Conv2d(out_channel, out_channel, 3, 1, 1),  # 3
#
#             nn.Conv2d(c3, out_channel, 1, 1, 0),  # 4
#             nn.Conv2d(out_channel, out_channel, 3, 1, 1),  # 5
#             nn.Conv2d(c5, out_channel, 3, 2, 1),  # 6
#             nn.ReLU(),  # 7
#             nn.Conv2d(out_channel, out_channel, 3, 2, 1),  # 8
#         )
#
#     def forward(self, x):
#         c3, c4, c5 = x
#         f5 = self.bones[0](c5)
#         p5 = self.bones[1](f5)
#
#         f4 = self.bones[2](c4) + nn.UpsamplingNearest2d(size=(c4.shape[2:]))(f5)
#         p4 = self.bones[3](f4)
#
#         f3 = self.bones[4](c3) + nn.UpsamplingNearest2d(size=(c3.shape[2:]))(f4)
#         p3 = self.bones[5](f3)
#
#         p6 = self.bones[6](c5)
#
#         p7 = self.bones[8](self.bones[7](p6))
#
#         return [p3, p4, p5, p6, p7]
