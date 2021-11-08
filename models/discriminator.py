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
    def __init__(self, num_classes, num_in_channels, num_features):
        super().__init__()

        self.main = nn.Sequential(
            # nn.AdaptiveAvgPool2d(64),
            nn.Conv2d(num_in_channels, num_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features * 2, 4, 1, 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 2, num_features * 4, 4, 1, 1, bias=False),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 4, num_features * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 8, 1, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Hardcode 64 layers (feature map output from main is 8x8)
        self.mlp = nn.Sequential(
            nn.Linear(64 + num_classes, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, input):
        images, one_hot_labels = input
        batch_size = images.shape[0]
        embeddings = self.main(images).view(batch_size, -1)
        conditional_embedding = torch.hstack((embeddings, one_hot_labels))
        probs = torch.sigmoid(self.mlp(conditional_embedding)).squeeze()
        return probs
