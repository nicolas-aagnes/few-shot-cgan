import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import torchvision.transforms as transforms

from datasets.noisy_mnist import NoisyMNIST
from models.generator import Generator
from models.discriminator import Discriminator
from models import utils


def main(args):
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cudnn.benchmark = True

    dataset = NoisyMNIST(
        1000,
        0.05,
        args.dataroot,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers)
    )

    device = torch.device("cuda:0" if args.cuda else "cpu")

    netG = Generator(dataset.num_channels, args.ngf).to(device)
    netG.apply(utils.weights_init)
    if args.netG != "":
        netG.load_state_dict(torch.load(args.netG))
    print(netG)

    netD = Discriminator(dataset.num_channels, args.ndf).to(device)
    netD.apply(utils.weights_init)
    if args.netD != "":
        netD.load_state_dict(torch.load(args.netD))
    print(netD)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(args.batch_size, args.nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    if args.dry_run:
        args.niter = 1

    for epoch in range(args.niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full(
                (batch_size,), real_label, dtype=real_cpu.dtype, device=device
            )

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print(
                "[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f"
                % (
                    epoch,
                    args.niter,
                    i,
                    len(dataloader),
                    errD.item(),
                    errG.item(),
                    D_x,
                    D_G_z1,
                    D_G_z2,
                )
            )
            if i % 100 == 0:
                vutils.save_image(
                    real_cpu, "%s/real_samples.png" % args.outf, normalize=True
                )
                fake = netG(fixed_noise)
                vutils.save_image(
                    fake.detach(),
                    "%s/fake_samples_epoch_%03d.png" % (args.outf, epoch),
                    normalize=True,
                )

            if args.dry_run:
                break
        # do checkpointing
        torch.save(netG.state_dict(), "%s/netG_epoch_%d.pth" % (opt.outf, epoch))
        torch.save(netD.state_dict(), "%s/netD_epoch_%d.pth" % (opt.outf, epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataroot", default="data", help="path to dataset")
    parser.add_argument(
        "--workers", default=1, type=int, help="number of data loading workers"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="input batch size")
    parser.add_argument(
        "--nz", type=int, default=100, help="size of the latent z vector"
    )
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument(
        "--niter", type=int, default=25, help="number of epochs to train for"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="learning rate, default=0.0002"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
    )
    parser.add_argument("--cuda", action="store_true", help="enables cuda")
    parser.add_argument(
        "--dry-run", action="store_true", help="check a single training cycle works"
    )
    parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
    parser.add_argument(
        "--netG", default="", help="path to netG (to continue training)"
    )
    parser.add_argument(
        "--netD", default="", help="path to netD (to continue training)"
    )
    parser.add_argument(
        "--outdir", default=".", help="folder to output images and model checkpoints"
    )
    parser.add_argument("--seed", type=int, help="manual seed")

    args = parser.parse_args()
    main(args)
