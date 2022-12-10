import torch.nn as nn
from ConvLSTM import ConvLSTM


class Net(nn.Module):

    def __init__(self, num_channels, num_kernels, kernel_size, padding, activation, frame_size, num_layers, ffc=False,
                 attention=False):
        super(Net, self).__init__()

        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1", ConvLSTM(
                in_channels=num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding,
                activation=activation, frame_size=frame_size, ffc=ffc, attention=attention)
        )

        self.sequential.add_module("batchnorm1", nn.BatchNorm3d(num_features=num_kernels))

        # Add rest of the layers
        for l in range(2, num_layers + 1):
            self.sequential.add_module(
                f"convlstm{l}", ConvLSTM(
                    in_channels=num_kernels, out_channels=num_kernels,
                    kernel_size=kernel_size, padding=padding,
                    activation=activation, frame_size=frame_size, ffc=False, attention=False)
            )

            self.sequential.add_module(f"batchnorm{l}", nn.BatchNorm3d(num_features=num_kernels))

            # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(in_channels=num_kernels, out_channels=num_channels, kernel_size=kernel_size,
                              padding=padding)

    def forward(self, x):
        # Forward propagation through all the layers
        output = self.sequential(x)

        # Return only the last output frame
        output = self.conv(output[:, :, -1])

        return nn.Sigmoid()(output)  # (batch_size, num_channels, height, width)
