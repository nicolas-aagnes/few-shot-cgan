"""Evaluate accuracy of a cGAN on MNIST with an oracle classifier."""
import argparse
from models.generator import ConditionalGenerator
import torch
import torchvision
import torch.nn.functional as F
from train_classifier import Net as MNISTClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    # Load oracle.
    oracle = MNISTClassifier()
    oracle.load_state_dict(torch.load(args.oracle, map_location=DEVICE))
    oracle.eval()

    # Load generator.
    netG = ConditionalGenerator(10, 100, 64, 1, 28)
    netG.load_state_dict(torch.load(args.netG, map_location=DEVICE))
    netG.eval()

    # Generate images.
    num_samples = args.num_samples // 10 * 10
    noise = torch.randn(num_samples, 100)
    labels = torch.arange(10).repeat_interleave(num_samples // 10)
    one_hot_labels = F.one_hot(labels)
    images = netG(torch.hstack((noise, one_hot_labels)))

    # Classify images.
    transform = torchvision.transforms.Normalize((0.5,), (0.5,))
    true_labels = oracle(transform(images))
    print(
        f"Accuracy {torch.sum(torch.argmax(true_labels, dim=1) == labels) / len(labels) * 100:.1f}%"
    )


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
