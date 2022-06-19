import logging
import os
import random

import numpy as np
import torch
import torch.backends.cudnn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import tqdm

logger = logging.getLogger('utils')

ave = 0.5               # average for normalization
std = 0.5               # std. for normalization
INF = 1e+30


def set_seed_num(seed_num):
    """set seed number.
    set seed number to python-random, numpy, torch (torch, cuda, backends), and os environ for reproductivity
    :param: seed number to set
    """
    if seed_num is None:
        seed_num = np.random.randint(0, (2 ** 30) - 1)
    np.random.seed(seed_num)
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed_num)
    with open('seed_num.csv', 'w') as f:
        print(seed_num, sep=',', file=f)
    return


def set_GPU():
    """get GPU settings and return.
    :return: obtained device
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info('using processor: {}'.format(device))
    return device


def load_data(batch_size, val_ratio):
    """load data and generate loader iterators.
    :param batch_size:
    :param val_ratio:
    :return: train, test, and val loader.
    """
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
    """generate loader with enumerate and also tqdm progress bar.
    :param loader:
    :param is_quiet:
    :return: loader
    """
    if is_quiet:
        enum_loader = enumerate(loader)
    else:
        enum_loader = enumerate(tqdm.tqdm(loader))
    return enum_loader
