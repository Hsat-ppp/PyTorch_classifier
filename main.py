import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

ave = 0.5               # average for normalization
std = 0.5               # std. for normalization
batch_size_train = 256  # batch size for training dataset
batch_size_test = 16    # batch size for test dataset
val_ratio = 0.2         # ratio of validation data from whole dataset
epoch_num = 10          # number of training epochs


class Net(nn.Module):
    # define network architecture
    def __init__(self):
        super(Net, self).__init__()
        self.init_conv = nn.Conv2d(3,16,3,padding=1)
        self.conv1 = nn.ModuleList([nn.Conv2d(16,16,3,padding=1) for _ in range(3)])
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(16) for _ in range(3)])
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.ModuleList([nn.Linear(16*16*16, 128), nn.Linear(128, 32)])
        self.output_fc = nn.Linear(32, 10)

    # forward calculation
    def forward(self, x):
        x = F.relu(self.init_conv(x))
        for l,bn in zip(self.conv1, self.bn1):
            x = F.relu(bn(l(x)))
        x = self.pool(x)
        x = x.view(-1,16*16*16) # flatten
        for l in self.fc1:
            x = F.relu(l(x))
        x = self.output_fc(x)
        return x


def set_GPU():
    # GPU settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    return device


def load_data():
    # load dataset
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((ave,),(std,))])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # split validation data
    n_samples = len(train_set)
    val_size = int(n_samples * val_ratio)
    train_set, val_set = torch.utils.data.random_split(train_set, [(n_samples-val_size), val_size])

    # generate data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size_train, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=False, num_workers=2)

    return train_loader, test_loader, val_loader


def train():
    # basic settings
    device = set_GPU()
    train_loader, test_loader, val_loader = load_data()
    model = Net()
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
            inputs, labels = data[0].to(device), data[1].to(device) # data は [inputs, labels] のリスト

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


if __name__ == "__main__":
    train()
