import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self, num_in_channels, num_features):
        super().__init__()

        self.main = nn.Sequential(
            nn.AdaptiveMaxPool2d(64),
            nn.Conv2d(num_in_channels, num_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 4, num_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 8, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_in_channels, num_features):
        super().__init__()

        self.main = nn.Sequential(
            # nn.AdaptiveAvgPool2d(28),
            nn.Conv2d(num_in_channels, num_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 4, num_features * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 8, 1, 4, 1, 1, bias=False),
            nn.Flatten(),
            nn.Sigmoid(),
        )

    def forward(self, input):
        probs = self.main(input).squeeze()
        return probs
