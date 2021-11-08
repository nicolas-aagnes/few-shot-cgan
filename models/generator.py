import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, num_z, num_features, num_out_channels, output_shape):
        super().__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(num_z, num_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_features * 8, num_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_features * 4, num_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_features * 2, num_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_features, num_out_channels, 4, 2, 1, bias=False),
            nn.AdaptiveAvgPool2d(output_shape),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)
