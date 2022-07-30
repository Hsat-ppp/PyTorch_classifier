import argparse
import json
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torchsummary

import PyTorch_classifier.model.CNN_based_model
import PyTorch_classifier.model.trainer
from PyTorch_classifier.utils.utils import load_data, set_GPU, set_seed_num

logger = logging.getLogger('controller')


def set_argparser_options():
    """set argparser options
    :return: ArgumentParser object
    """
    parser = argparse.ArgumentParser(description='''
                                    This is a CNN-based classifier code for CIFAR10.
                                    ''')
    parser.add_argument('-e', '--num_of_epochs', default=30, type=int,
                        help='number of training epochs')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        help='batch size.')
    parser.add_argument('-l', '--learning_rate', default=0.001, type=float,
                        help='initial learning rate.')
    parser.add_argument('-p', '--patience_for_lr_reducer', default=3, type=int,
                        help='patience epochs for learning rate reducer.')
    parser.add_argument('-r', '--ratio_of_validation_data', default=0.2, type=float,
                        help='ratio of validation data to entire training data.')
    parser.add_argument('-s', '--seed_num', type=int,
                        help='seed number for reproduction.')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='if set, we disables progress bar.')
    return parser


def check_args(args):
    """check args. if something is wrong, assert error will arise.
    :param args: args to be checked
    :return:
    """
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
    """evaluate a model.
    including:
        :parse args
        :obtain model from model folder
        :load data
        :define criterion and optimization strategy
        :run training
    :return:
    """
    # get args
    parser = set_argparser_options()
    args = parser.parse_args()
    check_args(args)
    # save args
    logger.info('args options')
    logger.info(args.__dict__)
    with open('args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    # set seed num
    set_seed_num(args.seed_num)

    # define model and trainer
    model = PyTorch_classifier.model.CNN_based_model.BasicCNNClassifier()
    torchsummary.summary(model, (3, 32, 32))
    trainer = PyTorch_classifier.model.trainer.ModelTrainer(model, 'history.csv', 'best.pth', True)

    # load data
    train_loader, test_loader, val_loader = load_data(args.batch_size, args.ratio_of_validation_data)
    trainer.set_loader(train_loader, test_loader, val_loader)

    # define device
    device = set_GPU()

    # set criterion
    trainer.set_criterion(nn.CrossEntropyLoss())
    trainer.set_optimizer(optim.Adam(model.parameters(), lr=args.learning_rate))

    # run training and test
    trainer.train(device, args.num_of_epochs, args.patience_for_lr_reducer, args.quiet)
    trainer.test(device, args.quiet)
