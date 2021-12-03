from __future__ import print_function
import argparse
import time
from models.utils import weights_init
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torchvision
import tqdm

from datasets.celeba import CelebADataset

POS_WEIGHTS = torch.FloatTensor(
    [
        7.9980,
        2.7456,
        0.9512,
        3.8883,
        43.5566,
        5.5974,
        3.1529,
        3.2638,
        3.1797,
        5.7571,
        18.6469,
        3.8734,
        6.0340,
        16.3711,
        20.4186,
        14.3566,
        14.9326,
        22.8380,
        1.5845,
        1.1976,
        1.3995,
        1.0686,
        23.0702,
        7.6844,
        0.1977,
        2.5194,
        22.2846,
        2.6043,
        11.5347,
        14.2158,
        16.6958,
        1.0743,
        3.7984,
        2.1292,
        4.2931,
        19.6355,
        1.1167,
        7.1323,
        12.7523,
        0.2926,
    ]
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = torchvision.models.mobilenet_v3_large(
            pretrained=True, progress=False
        )
        self.net.classifier = nn.Sequential(
            nn.Linear(in_features=960, out_features=1280, bias=True),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=40, bias=True),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.net = torchvision.models.mobilenet_v3_large(
            pretrained=True, progress=False
        )
        self.net.classifier = nn.Sequential(
            nn.Linear(in_features=960, out_features=1280, bias=True),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=512, bias=True),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=512, out_features=40, bias=True),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        start_time = time.time()
        data, target = data.to(device).float(), target.to(device).float()
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy_with_logits(
            output, target, pos_weight=POS_WEIGHTS.to(device)
        )
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tTime: {:0.3f}\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    time.time() - start_time,
                    loss.item(),
                )
            )
            if args.dry_run:
                break
        # if batch_idx == 1:
        #     break


def test(model, device, test_loader):
    model.eval()
    test_losses = []
    accuracies = []
    with torch.no_grad():
        pbar = tqdm.tqdm(test_loader)
        for data, target in pbar:
            data, target = data.to(device).float(), target.to(device).float()
            output = model(data)
            batch_loss = F.binary_cross_entropy_with_logits(
                output, target, reduction="mean"
            )
            test_losses.append(batch_loss)  # TODO: Check this.
            probs = torch.sigmoid(output)
            # assert probs.size() == (1000, 40), probs.size()
            accuracy = (probs > 0.5).int().eq(target.int()).float().mean(1)
            accuracies.append(accuracy)

            pbar.set_postfix(
                {"loss": batch_loss.item(), "accuracy": accuracy.mean().item()}
            )

    test_loss = torch.stack(test_losses).mean()
    accuracy = torch.cat(accuracies).mean() * 100

    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.1f}%")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=400,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset1 = CelebADataset(
        root="./data/celeba",
        split="train",
        target_type="attr",
        transform=transform,
        target_transform=None,
        download=False,
    )
    dataset2 = CelebADataset(
        root="./data/celeba",
        split="valid",
        target_type="attr",
        transform=transform,
        target_transform=None,
        download=False,
    )
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.008, weight_decay=0.00001)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(), "celeba_oracle2.pth")


if __name__ == "__main__":
    main()
