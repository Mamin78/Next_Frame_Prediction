import torch
import torch.nn as nn
from ConvLSTMCell import ConvLSTMCell

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvLSTM(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, activation, frame_size, ffc=False,
                 attention=False):
        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels, kernel_size, padding, activation, frame_size, ffc,
                                         attention)

    def forward(self, x):
        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, _, seq_len, height, width = x.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len, height, width, device=device)
        # output = torch.zeros(batch_size, self.out_channels, seq_len, height, width)

        # Initialize Hidden State
        hidden_state = torch.zeros(batch_size, self.out_channels, height, width, device=device)
        # hidden_state = torch.zeros(batch_size, self.out_channels, height, width)

        # Initialize Cell Input
        cell_input = torch.zeros(batch_size, self.out_channels, height, width, device=device)
        # cell_input = torch.zeros(batch_size, self.out_channels, height, width)

        # Unroll over time steps
        for time_step in range(seq_len):
            hidden_state, cell_input = self.convLSTMcell(x[:, :, time_step], hidden_state, cell_input)

            output[:, :, time_step] = hidden_state

        return output
