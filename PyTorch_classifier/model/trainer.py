import logging

import numpy as np
import torch
import torch.optim as optim

from PyTorch_classifier.utils.utils import make_enum_loader, INF

logger = logging.getLogger('trainer')


class ModelTrainer(object):
    """
    model training class.
    manage training via set loader, criterion, and optimizer.
    """
    def __init__(self, model, device, history_file_name, output_model_file_name):
        """init function.
        :param model:
        :param device:
        :param history_file_name:
        :param output_model_file_name:
        """
        self.model = model
        self.device = device
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self.history_file_name = history_file_name
        self.output_model_file_name = output_model_file_name
        self.criterion = None
        self.optimizer = None
        self.best_loss_value = INF
        self.test_loss_value = INF

    def set_loader(self, train_loader, test_loader, val_loader):
        """set data loader
        :param train_loader:
        :param test_loader:
        :param val_loader:
        :return:
        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

    def set_criterion(self, criterion):
        """set criterion
        :param criterion:
        :return:
        """
        self.criterion = criterion

    def set_optimizer(self, optimizer):
        """set optimizer.
        :param optimizer:
        :return:
        """
        self.optimizer = optimizer

    def calculate_loss(self, inputs, labels):
        """calculate loss. please change this properly if criterion is changed.
        :param inputs:
        :param labels:
        :return:
        """
        outputs = self.model(inputs)  # forward calculation
        loss = self.criterion(outputs, labels)  # calculate loss
        return loss

    def train(self, num_of_epochs, patience, is_quiet=False):
        """run training.
        :param num_of_epochs:
        :param patience:
        :param is_quiet:
        :return:
        """
        # settings
        self.model.to(self.device)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience, verbose=True)
        with open("history.csv", 'w'):
            pass

        # training process
        self.best_loss_value = INF
        logger.info('training start')
        for epoch in range(num_of_epochs):
            train_loss = 0.0
            val_loss = 0.0
            train_batches = 0
            val_batches = 0
            logger.info('training phase in epoch {}'.format(epoch + 1))
            self.model.train()  # train mode
            enum_loader = make_enum_loader(self.train_loader, is_quiet)
            for i, data in enum_loader:  # load every batch
                inputs, labels = data[0].to(self.device), data[1].to(self.device)  # data は [inputs, labels] のリスト
                # reset gradients
                self.optimizer.zero_grad()
                # calculation
                loss = self.calculate_loss(inputs, labels)
                # accumulate loss
                train_loss += loss.item()
                train_batches += 1
                # update
                loss.backward()  # backpropagation
                self.optimizer.step()  # update parameters

            # validation loss calculation
            self.model.eval()  # evaluation mode
            logger.info('validation phase in epoch {}'.format(epoch + 1))
            enum_loader = make_enum_loader(self.val_loader, is_quiet)
            with torch.no_grad():
                for i, data in enum_loader:  # load every batch
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)  # data は [inputs, labels] のリスト
                    loss = self.calculate_loss(inputs, labels)
                    # accumulate loss
                    val_loss += loss.item()
                    val_batches += 1

            # output history
            logger.info('epoch {0} train_loss: {1}'.format(epoch + 1, train_loss / train_batches))
            logger.info('epoch {0} val_loss: {1}'.format(epoch + 1, val_loss / val_batches))
            with open("history.csv", 'a') as f:
                print(epoch + 1, train_loss / train_batches, val_loss / val_batches, sep=',', file=f)

            # save the best model
            if self.best_loss_value > val_loss / val_batches:
                self.best_loss_value = val_loss / val_batches
                PATH = self.output_model_file_name
                torch.save(self.model.state_dict(), PATH)

            # update learning rate
            scheduler.step(val_loss / val_batches)

        logger.info('training end')

    def test(self, is_quiet, pre_trained_file=None):
        """run test. this need pre-trained weight file.
        :param is_quiet:
        :param pre_trained_file:
        :return:
        """
        if pre_trained_file is None:
            pre_trained_file = self.output_model_file_name
        # model setting
        self.model.load_state_dict(torch.load(pre_trained_file))
        self.model.eval()  # evaluation mode
        self.model.to(self.device)

        test_loss = 0.0
        test_batches = 0.0
        logger.info('running test...')

        # test
        enum_loader = make_enum_loader(self.test_loader, is_quiet)
        ground_truth = []
        predicted = []
        with torch.no_grad():
            for i, data in enum_loader:  # load every batch
                inputs, labels = data[0].to(self.device), data[1].to(self.device)  # data は [inputs, labels] のリスト
                loss = self.calculate_loss(inputs, labels)
                # accumulate loss
                test_loss += loss.item()
                test_batches += 1
                # preserve labels
                ground_truth.append(labels.detach().to('cpu').clone())
                outputs = self.model(inputs)
                predicted.append(outputs.detach().to('cpu').clone())

        logger.info('test loss: {}'.format(test_loss / test_batches))
        self.best_loss_value = test_loss / test_batches

        # show prediction
        ground_truth = torch.cat(ground_truth)
        predicted = torch.cat(predicted)
        _, predicted = torch.max(predicted, dim=1)
        accuracy = torch.sum(ground_truth == predicted) / ground_truth.shape[0]
        logger.info('accuracy: {}%'.format(accuracy * 100.0))
        logger.info('test end')
