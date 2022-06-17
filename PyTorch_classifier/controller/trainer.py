import torch
import torch.nn as nn
import torch.optim as optim

from PyTorch_classifier.utils.utils import load_data, set_GPU
from PyTorch_classifier.model.CNN_based_model import BasicCNNClassifier

epoch_num = 10          # number of training epochs


def train():
    # basic settings
    device = set_GPU()
    train_loader, test_loader, val_loader = load_data()
    model = BasicCNNClassifier()
    model.to(device)

    # criterion, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    # training process
    min_loss = 999999999
    print("training start")
    for epoch in range(epoch_num):
        train_loss = 0.0
        val_loss = 0.0
        train_batches = 0
        val_batches = 0
        model.train()   # train mode
        for i, data in enumerate(train_loader):   # load every batch
            inputs, labels = data[0].to(device), data[1].to(device)  # data は [inputs, labels] のリスト

            # reset gradients
            optimizer.zero_grad()

            outputs = model(inputs)    # forward calculation
            loss = criterion(outputs, labels)   # calculate loss
            loss.backward()                     # backpropagation
            optimizer.step()                    # update parameters

            # accumulate loss
            train_loss += loss.item()
            train_batches += 1

        # validation loss calculation
        model.eval()    # evaluation mode
        with torch.no_grad():
            for i, data in enumerate(val_loader):   # load every batch
                inputs, labels = data[0].to(device), data[1].to(device) # data は [inputs, labels] のリスト
                outputs = model(inputs)               # forward calculation
                loss = criterion(outputs, labels)   # calculate loss

                # accumulate loss
                val_loss += loss.item()
                val_batches += 1

        # output history
        print('epoch %d train_loss: %.10f' %
              (epoch + 1,  train_loss/train_batches))
        print('epoch %d val_loss: %.10f' %
              (epoch + 1,  val_loss/val_batches))

        with open("history.csv",'a') as f:
            print(str(epoch+1) + ',' + str(train_loss/train_batches) + ',' + str(val_loss/val_batches),file=f)

        # save the best model
        if min_loss > val_loss/val_batches:
            min_loss = val_loss/val_batches
            PATH = "best.pth"
            torch.save(model.state_dict(), PATH)

        # update learning rate
        scheduler.step(val_loss/val_batches)

    # save the latest model
    print("training finished")
    PATH = "lastepoch.pth"
    torch.save(model.state_dict(), PATH)
