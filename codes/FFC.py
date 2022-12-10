import torch
import torch.nn as nn


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                    kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        # print(x.size())
        # fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        # print(ffted.size())

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4,
                                                                       2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        # ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim)

        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, enable_lfu=True):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu

        if stride == 2:
            self.down_sample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.down_sample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

        self.fu = FourierUnit(out_channels // 2, out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        # print("input x size at SpectralTransform", x.size())
        x = self.down_sample(x)
        x = self.conv1(x)
        output = self.fu(x)
        output = self.conv2(x + output)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, bias=False):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        self.in_cg = in_cg
        self.in_cl = in_cl

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        self.conv_l2l = nn.Conv2d(in_cl, out_cl, kernel_size, stride=stride, padding=padding, dilation=dilation,
                                  bias=bias)
        self.conv_l2g = nn.Conv2d(in_cl, out_cg, kernel_size, stride=stride, padding=padding, dilation=dilation,
                                  bias=bias)
        self.conv_g2l = nn.Conv2d(in_cg, out_cl, kernel_size, stride=stride, padding=padding, dilation=dilation,
                                  bias=bias)

        self.conv_g2g = SpectralTransform(in_cg, out_cg, stride)

    def forward(self, x):
        # x_l, x_g = x if type(x) is tuple else (x, 0)
        x_l, x_g = x if type(x) is tuple else (x[:, :self.in_cl, ...], x[:, self.in_cl:, ...])
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.conv_l2l(x_l) + self.conv_g2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.conv_l2g(x_l) + self.conv_g2g(x_g)

        return out_xl, out_xg


class FfcBnAct(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 merge=True):
        super(FfcBnAct, self).__init__()

        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       bias)

        self.bn_l = norm_layer(int(out_channels * (1 - ratio_gout)))
        self.bn_g = norm_layer(int(out_channels * ratio_gout))

        self.act_l = activation_layer(inplace=True)
        self.act_g = activation_layer(inplace=True)
        self.merge = merge

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))

        if self.merge:
            return torch.cat((x_l, x_g), dim=1)
        else:
            return x_l, x_g


# if __name__ == '__main__':
#     f = FfcBnAct(1 + 32, 4 * 32,
#                  kernel_size=(3, 3),
#                  padding=(1, 1),
#                  ratio_gin=0.5, ratio_gout=0.5,
#                  activation_layer=nn.ReLU)
