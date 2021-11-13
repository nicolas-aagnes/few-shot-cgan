import torch
import torchvision.transforms as transforms


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def prepare_data_for_inception(x, device):
    assert len(x.shape) == 4

    # Change from graysscale to RGB.
    if x.shape[1] == 1:
        x = x.expand(x.shape[0], 3, x.shape[2], x.shape[3])

    x = torch.nn.functional.interpolate(x, 299, mode="bicubic", align_corners=False)
    minv, maxv = float(x.min()), float(x.max())
    x.clamp_(min=minv, max=maxv).add_(-minv).div_(maxv - minv + 1e-5)
    x.mul_(255).add_(0.5).clamp_(0, 255)

    return x.to(device).to(torch.uint8)


def get_mnist_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
