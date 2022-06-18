import argparse
import logging

import torch
import torchvision
import torchvision.transforms as transforms
import tqdm

logger = logging.getLogger('info_logger')

ave = 0.5               # average for normalization
std = 0.5               # std. for normalization


def get_argparser_options():
    parser = argparse.ArgumentParser(description='''
                                    This is a CNN-based classifier code for CIFAR10.
                                    ''')
    parser.add_argument('-e', '--num_of_epochs', default=15, type=int,
                        help='number of training epochs')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        help='batch size.')
    parser.add_argument('-l', '--learning_rate', default=0.001, type=float,
                        help='initial learning rate.')
    parser.add_argument('-p', '--patience_for_lr_reducer', default=5, type=int,
                        help='patience epochs for learning rate reducer.')
    parser.add_argument('-r', '--ratio_of_validation_data', default=0.2, type=float,
                        help='ratio of validation data to entire training data.')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='if set, we disables progress bar.')
    args = parser.parse_args()
    return args


def check_args(args):
    assert args.num_of_epochs >= 1, 'Option "num_of_epochs" need to be positive. ' \
                                    'Got: {}'.format(args.num_of_epochs)
    assert args.batch_size >= 1, 'Option "batch_size" need to be positive. ' \
                                 'Got: {}'.format(args.batch_size)
    assert 0.0 < args.learning_rate < 1.0, 'Option "learning_rate" need to be between 0.0 to 1.0. ' \
                                           'Got: {}'.format(args.learning_rate)
    assert 1 <= args.patience_for_lr_reducer <= args.num_of_epochs, \
        'Option "patience_for_lr_reducer" need to be between 1 to "num_of_epochs". ' \
        'Got: {}'.format(args.batch_size)
    assert 0.0 < args.ratio_of_validation_data < 1.0, \
        'Option "ratio_of_validation_data" need to be between 0.0 to 1.0. ' \
        'Got: {}'.format(args.ratio_of_validation_data)


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
