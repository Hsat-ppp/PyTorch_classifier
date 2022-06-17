import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ave = 0.5               # average for normalization
std = 0.5               # std. for normalization
batch_size_train = 256  # batch size for training dataset
batch_size_test = 16    # batch size for test dataset
val_ratio = 0.2         # ratio of validation data from whole dataset


def set_GPU():
    # GPU settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    return device


def load_data():
    # load dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((ave,), (std,))])
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
