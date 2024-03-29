"""Evaluate accuracy of a cGAN on MNIST with an oracle classifier."""
import argparse
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torchmetrics
import tqdm
from torchvision.datasets.mnist import MNIST

from models.generator import ConditionalGenerator
from train_classifier import Net as MNISTClassifier
from models import utils


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    # Test on batches of 1000 images.
    num_batch_samples = 1000
    assert args.num_samples % num_batch_samples == 0

    # Load oracle.
    oracle = MNISTClassifier()
    oracle.load_state_dict(torch.load(args.oracle, map_location=DEVICE))
    oracle.eval()

    # Load generator.
    netG = ConditionalGenerator(10, 100, 64, 1, 28)
    netG.load_state_dict(torch.load(args.netG, map_location=DEVICE))
    netG.eval()

    # Dataset with real images.
    mnist = MNIST("data", train=True, download=True)

    # Metrics.
    accuracy = torchmetrics.Accuracy().to(DEVICE)
    is_ = torchmetrics.IS().to(DEVICE)
    fid = torchmetrics.FID().to(DEVICE)
    kid = torchmetrics.KID().to(DEVICE)

    with torch.no_grad():
        for _ in tqdm.tqdm(range(args.num_samples // num_batch_samples)):
            # Get random real images.
            real_images = torch.stack(
                [
                    torchvision.transforms.PILToTensor()(mnist[index][0]).float()
                    for index in np.random.choice(
                        len(mnist), num_batch_samples, replace=False
                    )
                ]
            )
            real_images = utils.prepare_data_for_inception(real_images)

            # Generate images.
            noise = torch.randn(num_batch_samples, 100)
            labels = torch.arange(10).repeat_interleave(num_batch_samples // 10)
            one_hot_labels = F.one_hot(labels)
            fake_images = netG(torch.hstack((noise, one_hot_labels)))

            # Classify images.
            transform = torchvision.transforms.Normalize((0.5,), (0.5,))
            true_labels = oracle(transform(fake_images))

            # Compute metrics.
            fake_images = utils.prepare_data_for_inception(fake_images)
            accuracy.update(labels, torch.argmax(true_labels, dim=1))
            is_.update(fake_images)
            fid.update(real_images, real=True)
            fid.update(fake_images, real=False)
            kid.update(real_images, real=True)
            kid.update(fake_images, real=False)

    print(f"Accuracy {accuracy.compute()*100:.1f}%")
    print(f"IS {is_.compute()[0].item()}")
    print(f"FID {fid.compute().item()}")
    print(f"KID {kid.compute()[0].item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", help="Path to oracle classifier model")
    parser.add_argument("--netG", help="Path to generator network checkpoint")
    parser.add_argument(
        "--num-samples",
        default=10000,
        type=int,
        help="Number of images to generate for testing",
    )
    args = parser.parse_args()
    main(args)
