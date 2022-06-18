import logging

import torch
import torch.optim as optim

from PyTorch_classifier.utils.utils import make_enum_loader

logger = logging.getLogger('trainer')


class ModelTrainer(object):
    def __init__(self, model, history_file_name, output_model_file_name):
        self.model = model
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self.history_file_name = history_file_name
        self.output_model_file_name = output_model_file_name

    def set_loader(self, train_loader, test_loader, val_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

    def calculate_loss(self, inputs, labels, criterion):
        outputs = self.model(inputs)  # forward calculation
        loss = criterion(outputs, labels)  # calculate loss
        return loss

    def train(self, device, criterion, optimizer, num_of_epochs, patience, is_quiet=False):
        # settings
        self.model.to(device)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, verbose=True)
        with open("history.csv", 'w'):
            pass

        # training process
        min_loss = 1E+30
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
                inputs, labels = data[0].to(device), data[1].to(device)  # data は [inputs, labels] のリスト
                # reset gradients
                optimizer.zero_grad()
                # calculation
                loss = self.calculate_loss(inputs, labels, criterion)
                # accumulate loss
                train_loss += loss.item()
                train_batches += 1
                # update
                loss.backward()  # backpropagation
                optimizer.step()  # update parameters

            # validation loss calculation
            self.model.eval()  # evaluation mode
            logger.info('validation phase in epoch {}'.format(epoch + 1))
            enum_loader = make_enum_loader(self.val_loader, is_quiet)
            with torch.no_grad():
                for i, data in enum_loader:  # load every batch
                    inputs, labels = data[0].to(device), data[1].to(device)  # data は [inputs, labels] のリスト
                    loss = self.calculate_loss(inputs, labels, criterion)
                    # accumulate loss
                    val_loss += loss.item()
                    val_batches += 1

            # output history
            logger.info('epoch {0} train_loss: {1}'.format(epoch + 1, train_loss / train_batches))
            logger.info('epoch {0} val_loss: {1}'.format(epoch + 1, val_loss / val_batches))
            with open("history.csv", 'a') as f:
                print(epoch + 1, train_loss / train_batches, val_loss / val_batches, sep=',', file=f)

            # save the best model
            if min_loss > val_loss / val_batches:
                min_loss = val_loss / val_batches
                PATH = self.output_model_file_name
                torch.save(self.model.state_dict(), PATH)

            # update learning rate
            scheduler.step(val_loss / val_batches)

        logger.info('training end')

    def test(self, device, criterion, is_quiet, pre_trained_file=None):
        if pre_trained_file is None:
            pre_trained_file = self.output_model_file_name
        # model setting
        self.model.load_state_dict(torch.load(pre_trained_file))
        self.model.eval()  # evaluation mode
        self.model.to(device)

        test_loss = 0.0
        test_batches = 0.0
        logger.info('running test...')

        # test
        enum_loader = make_enum_loader(self.test_loader, is_quiet)
        with torch.no_grad():
            for i, data in enum_loader:  # load every batch
                inputs, labels = data[0].to(device), data[1].to(device)  # data は [inputs, labels] のリスト
                loss = self.calculate_loss(inputs, labels, criterion)
                # accumulate loss
                test_loss += loss.item()
                test_batches += 1

        logger.info('test loss: {}'.format(test_loss / test_batches))
        logger.info('test end')
