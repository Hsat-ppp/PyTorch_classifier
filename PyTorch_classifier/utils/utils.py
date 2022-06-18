import logging

import torch
import torchvision
import torchvision.transforms as transforms
import tqdm

logger = logging.getLogger('info_logger')

ave = 0.5               # average for normalization
std = 0.5               # std. for normalization


def set_GPU():
    # GPU settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info('using processor: {}'.format(device))
    return device


def load_data(batch_size, val_ratio):
    # load dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((ave,), (std,))])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # split validation data
    n_samples = len(train_set)
    val_size = int(n_samples * val_ratio)
    train_set, val_set = torch.utils.data.random_split(train_set, [(n_samples-val_size), val_size])

    # generate data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, val_loader


def make_enum_loader(loader, is_quiet):
    if is_quiet:
        enum_loader = enumerate(loader)
    else:
        enum_loader = enumerate(tqdm.tqdm(loader))
    return enum_loader
