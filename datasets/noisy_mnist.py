from torch import distributions, dtype
from torchvision.datasets.mnist import MNIST
from typing import Optional, Callable
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.distributions.categorical import Categorical


class NoisyMNIST(MNIST):
    # TODO: Extend noise distribution to more than just the uniform distribution.

    def __init__(
        self,
        dataset_size: int,
        noise_level: float,
        root: str,
        train=True,
        entropy=None,
    ):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        super().__init__(
            root=root,
            train=train,
            transform=transform,
            download=True,
        )

        # MNIST images are greyscale.
        self.num_channels = 1
        self.output_shape = 28
        self.num_classes = 10

        assert self.data.shape[-2:] == (28, 28)

        assert (
            dataset_size <= 50000
        ), "The MNIST dataset is slightly unbalanced so the max is set to 50k instead of 60k"

        dataset_size = dataset_size // 10 * 10

        # Get all indices with label 1. Select num_per_class elements of that.
        num_images_per_class = dataset_size // 10

        images, targets = [], []
        for i in range(10):
            indices = np.random.choice(
                (self.targets == i).nonzero().squeeze(),
                num_images_per_class,
                replace=False,
            )
            images.append(self.data[indices])
            targets.append(self.targets[indices])
        self.data, self.targets = torch.cat(images), torch.cat(targets)

        # Uniform categorical distribution.
        distribution = Categorical(torch.ones(10))
        max_entropy = distribution.entropy()

        # Compute non-uniform distribution.
        if entropy is not None:
            assert 0.0 < entropy < max_entropy

            curr_entropy = -1
            while abs(entropy - curr_entropy) > 0.01 * max_entropy:
                distribution = Categorical(torch.rand(10))
                curr_entropy = distribution.entropy()

        # Substitute with noisy labels.
        num_noisy_targets = int(dataset_size * noise_level)
        indices = np.random.choice(dataset_size, num_noisy_targets, replace=False)
        noisy_targets = np.random.choice(
            10, num_noisy_targets, replace=True, p=distribution.probs.numpy()
        ).astype(np.int64)
        self.targets[indices] = torch.from_numpy(noisy_targets)

        assert dataset_size == self.data.shape[0] == self.targets.shape[0] == len(self)
