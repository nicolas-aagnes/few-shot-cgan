from torchvision.datasets.mnist import MNIST
from typing import Optional, Callable
import torch
import numpy as np
import torchvision.transforms as transforms


class NoisyMNIST(MNIST):
    # TODO: Extend noise distribution to more than just the uniform distribution.

    def __init__(
        self,
        dataset_size: int,
        noise_level: float,
        root: str,
    ):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        super().__init__(
            root=root,
            train=True,
            transform=transform,
            download=True,
        )

        # MNIST images are greyscale.
        self.num_channels = 1
        self.output_shape = 28
        self.num_classes = 10

        assert self.data.shape[-2:] == (28, 28)

        assert dataset_size <= 60000, "There are only 60k images in the MNIST dataset."

        # Select dataset of correct size.
        self.data = self.data[:dataset_size]
        self.targets = self.targets[:dataset_size]

        # Sort the dataset.
        self.targets, indices = torch.sort(self.targets)
        self.data = self.data[indices]

        # Create noisy targets.
        num_noisy_targets = int(dataset_size * noise_level)
        indices = np.random.choice(dataset_size, num_noisy_targets, replace=False)
        noisy_targets = np.random.choice(10, num_noisy_targets, replace=True).astype(
            np.int64
        )
        self.targets[indices] = torch.from_numpy(noisy_targets)

        assert dataset_size == self.data.shape[0] == self.targets.shape[0] == len(self)
