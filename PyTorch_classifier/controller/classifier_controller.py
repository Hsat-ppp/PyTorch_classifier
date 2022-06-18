import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from PyTorch_classifier.model.CNN_based_model import BasicCNNClassifier
import PyTorch_classifier.model.trainer
from PyTorch_classifier.utils.utils import check_args, get_argparser_options, load_data, set_GPU


def evaluate_model():
    # get args
    args = get_argparser_options()
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
    trainer.train(args, device, criterion, optimizer, enable_scheduler=True)
    trainer.test(args, device, criterion, 'best.pth')
