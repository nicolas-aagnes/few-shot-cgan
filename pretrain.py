"""Script for pretraining the generator and discriminator networks."""
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from pathlib import Path

from datasets.noisy_mnist import NoisyMNIST
from models.generator import ConditionalGenerator
from models.discriminator import ConditionalDiscriminator
from models import utils


def main(args):
    if args.logdir is None:
        args.logdir = f"./pretrain/dataset_size={args.dataset_size},noise_level={args.noise_level}"
        args.logdir = "runs"
        Path(args.logdir).mkdir(exist_ok=True, parents=True)

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cudnn.benchmark = True

    dataset = NoisyMNIST(args.dataset_size, args.noise_level, args.dataroot)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    # Conditional Generator Network,
    netG = ConditionalGenerator(
        dataset.num_classes,
        args.nz,
        args.ngf,
        dataset.num_channels,
        dataset.output_shape,
    ).to(device)
    netG.apply(utils.weights_init)
    if args.netG != "":
        netG.load_state_dict(torch.load(args.netG))

    # Conditional Discriminator Network.
    netD = ConditionalDiscriminator(
        dataset.num_channels + dataset.num_classes, args.ndf
    ).to(device)
    netD.apply(utils.weights_init)
    if args.netD != "":
        netD.load_state_dict(torch.load(args.netD))

    # Tensorboard writer.
    writer = SummaryWriter(log_dir=args.logdir, flush_secs=10)

    # GAN loss and labels.
    criterion = nn.BCELoss()
    real_label = 1
    fake_label = 0

    # Setup optimizers.
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # Create fixed noise for evalutation images.
    noise = torch.randn(100, args.nz, device=device)
    labels = torch.arange(10).repeat_interleave(10).to(device)
    one_hot_labels = torch.nn.functional.one_hot(labels).to(device)
    fixed_fake_conditional_noise = torch.hstack((noise, one_hot_labels))

    if args.dry_run:
        args.niter = 1

    for epoch in range(1, args.niter + 1):
        for i_step, (real_images, labels) in enumerate(dataloader):
            # Create one hot labels for generator (one_hot_labels) and discriminator (image_one_hot_labels).
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=10).to(
                device
            )
            image_one_hot_labels = one_hot_labels.clone()[..., None, None].expand(
                -1, -1, 28, 28
            )
            #############################################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            #############################################################
            # Rrain with real images.
            netD.zero_grad()
            real_images = real_images.to(device)
            batch_size = real_images.shape[0]
            label = torch.full(
                (batch_size,), real_label, dtype=real_images.dtype, device=device
            )

            output = netD(torch.hstack((real_images, image_one_hot_labels)))
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, args.nz, device=device)
            conditional_noise = torch.hstack((noise, one_hot_labels))
            fake = netG(conditional_noise)
            label.fill_(fake_label)
            output = netD(torch.hstack((fake.detach(), image_one_hot_labels)))
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            #############################################################
            # (2) Update G network: maximize log(D(G(z)))
            #############################################################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(torch.hstack((fake, image_one_hot_labels)))
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Log to tensorboard.
            current_iter = (epoch - 1) * len(dataloader) + i_step
            writer.add_scalar("Loss/D", errD.item(), current_iter)
            writer.add_scalar("Loss/G", errG.item(), current_iter)
            writer.add_scalar("Probability/D(x)", D_x, current_iter)
            writer.add_scalar("Probability/D(G(z_1))", D_G_z1, current_iter)
            writer.add_scalar("Probability/D(G(z_2))", D_G_z2, current_iter)

            # Save model with visual images.
            if i_step % args.save_frequency == 0:
                print(
                    f"[{epoch}/{args.niter}][{i_step}/{len(dataloader)}] Loss D: {errD.item():.4f}"
                    + f" Loss_G {errG.item():.4f} D(x): {D_x:.4f} D(g(z)): {D_G_z1:.4f} / {D_G_z2:.4f}"
                )

                path = Path(args.logdir).joinpath(f"epoch{epoch}/iteration{i_step}")
                path.mkdir(exist_ok=True, parents=True)

                fake = netG(fixed_fake_conditional_noise)

                vutils.save_image(
                    fake.detach(),
                    f"{path}/iteration{i_step}.png",
                    nrow=10,
                    normalize=True,
                )

                # Do checkpointing.
                torch.save(netG.state_dict(), f"{path}/netG.pth")
                torch.save(netD.state_dict(), f"{path}/netD.pth")

            if args.dry_run:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataroot", default="data", help="path to dataset")
    parser.add_argument(
        "--dataset-size", default=50000, type=int, help="Number of real images to use"
    )
    parser.add_argument(
        "--noise-level",
        default=0.0,
        type=float,
        help="Percentage of labels to randomize",
    )
    parser.add_argument(
        "--num-workers", default=1, type=int, help="number of data loading workers"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="input batch size")
    parser.add_argument(
        "--nz", type=int, default=100, help="size of the latent z vector"
    )
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument(
        "--niter", type=int, default=5, help="number of epochs to train for"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="learning rate, default=0.0002"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="check a single training cycle works"
    )
    parser.add_argument(
        "--netG", default="", help="path to netG (to continue training)"
    )
    parser.add_argument(
        "--netD", default="", help="path to netD (to continue training)"
    )
    parser.add_argument(
        "--logdir", default=None, help="folder to output images and model checkpoints"
    )
    parser.add_argument(
        "--save-frequency", default=50, type=int, help="number of batches between saves"
    )
    parser.add_argument("--seed", type=int, help="manual seed")

    args = parser.parse_args()
    main(args)
