import torch
import torch.nn as nn
from SelfAttention import SelfAttention
from FFC import FfcBnAct


# Original ConvLSTM cell as proposed by Shi et al.
class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, activation, frame_size, ffc=False,
                 attention=False):

        super(ConvLSTMCell, self).__init__()

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        if ffc:
            self.conv = FfcBnAct(in_channels + out_channels, 4 * out_channels,
                                 kernel_size=kernel_size,
                                 padding=padding,
                                 stride=1,
                                 ratio_gin=0.5, ratio_gout=0.5,
                                 activation_layer=nn.ReLU,
                                 enable_lfu=False,
                                 merge=True)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels + out_channels, out_channels=4 * out_channels,
                                  kernel_size=kernel_size, padding=padding)

        self.use_attention = attention
        self.attention = SelfAttention(in_dim=4 * out_channels)

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

    def forward(self, x, hidden_prev, cell_prev):

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv(torch.cat([x, hidden_prev], dim=1))

        if self.use_attention:
            conv_output = self.attention(conv_output)

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        i_conv, f_conv, c_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * cell_prev)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * cell_prev)

        # Current Cell output
        cell_input = forget_gate * cell_prev + input_gate * self.activation(c_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * cell_input)

        # Current Hidden State
        hidden_state = output_gate * self.activation(cell_input)

        return hidden_state, cell_input
