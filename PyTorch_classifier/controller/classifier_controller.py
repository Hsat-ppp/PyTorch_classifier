import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from PyTorch_classifier.model.CNN_based_model import BasicCNNClassifier
import PyTorch_classifier.model.trainer
from PyTorch_classifier.utils.utils import load_data, set_GPU


def set_argparser_options():
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
    return parser


def check_args(args):
    assert args.num_of_epochs >= 1, 'Option "num_of_epochs" need to be positive. ' \
                                    'Got: {}'.format(args.num_of_epochs)
    assert args.batch_size >= 1, 'Option "batch_size" need to be positive. ' \
                                 'Got: {}'.format(args.batch_size)
    assert 0.0 < args.learning_rate < 1.0, 'Option "learning_rate" need to be between 0.0 to 1.0. ' \
                                           'Got: {}'.format(args.learning_rate)
    assert 1 <= args.patience_for_lr_reducer <= args.num_of_epochs, \
        'Option "patience_for_lr_reducer" need to be between 1 to "num_of_epochs". ' \
        'Got: {}'.format(args.patience_for_lr_reducer)
    assert 0.0 < args.ratio_of_validation_data < 1.0, \
        'Option "ratio_of_validation_data" need to be between 0.0 to 1.0. ' \
        'Got: {}'.format(args.ratio_of_validation_data)


def evaluate_model():
    # get args
    parser = set_argparser_options()
    args = parser.parse_args()
    check_args(args)

    # define model and trainer
    model = BasicCNNClassifier()
    trainer = PyTorch_classifier.model.trainer.ModelTrainer(model, 'history.csv', 'best.pth')

    # load data
    train_loader, test_loader, val_loader = load_data(args.batch_size, args.ratio_of_validation_data)
    trainer.set_loader(train_loader, test_loader, val_loader)

    # define device and model to be evaluated
    device = set_GPU()

    # set criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # run training and test
    trainer.train(device, criterion, optimizer, args.num_of_epochs, args.patience_for_lr_reducer, args.quiet)
    trainer.test(device, criterion, args.quiet)
