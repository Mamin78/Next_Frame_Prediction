import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation=None):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=(1, 1))
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=(1, 1))
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(1, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps : (batch_size, channels, height, width)
        returns :
            out : self attention value + input feature : (batch_size, channels, height, width)
        """
        batch_size, c, height, width = x.size()
        # print(x.size())
        proj_query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # (b, w*h, c//8)
        proj_key = self.key_conv(x).view(batch_size, -1, height * width)  # (b, c//8, w*h)
        energy = torch.bmm(proj_query, proj_key)  # (b, w*h, w*h)
        attention = self.softmax(energy)  # (b, w*h, w*h)
        proj_value = self.value_conv(x).view(batch_size, -1, height * width)  # (b, c, w*h)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # (b, c, w*h)
        out = out.view(batch_size, c, height, width)  # (b, c, h, w)
        out = self.gamma * out + x
        return out
