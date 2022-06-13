import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

ave = 0.5               # 正規化平均
std = 0.5               # 正規化標準偏差
batch_size_train = 256  # 学習バッチサイズ
batch_size_test = 16    # テストバッチサイズ
val_ratio = 0.2         # データ全体に対する検証データの割合
epoch_num = 10          # 学習エポック数

class Net(nn.Module):
    # ネットワーク構造の定義
    def __init__(self):
        super(Net, self).__init__()
        self.init_conv = nn.Conv2d(3,16,3,padding=1)
        self.conv1 = nn.ModuleList([nn.Conv2d(16,16,3,padding=1) for _ in range(3)])
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(16) for _ in range(3)])
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.ModuleList([nn.Linear(16*16*16, 128), nn.Linear(128, 32)])
        self.output_fc = nn.Linear(32, 10)

    # 順方向計算
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
    # GPUの設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    return device

def load_data():
    # データのロード
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((ave,),(std,))])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # 検証データのsplit
    n_samples = len(train_set)
    val_size = int(n_samples * val_ratio)
    train_set, val_set = torch.utils.data.random_split(train_set, [(n_samples-val_size), val_size])

    # DataLoaderの定義
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size_train, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=False, num_workers=2)

    return train_loader, test_loader, val_loader

def train():
    device = set_GPU()
    train_loader, test_loader, val_loader = load_data()
    model = Net()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    min_loss = 999999999
    print("training start")
    for epoch in range(epoch_num):
        train_loss = 0.0
        val_loss = 0.0
        train_batches = 0
        val_batches = 0
        model.train()   # 訓練モード
        for i, data in enumerate(train_loader):   # バッチ毎に読み込む
            inputs, labels = data[0].to(device), data[1].to(device) # data は [inputs, labels] のリスト

            # 勾配のリセット
            optimizer.zero_grad()

            outputs = model(inputs)    # 順方向計算
            loss = criterion(outputs, labels)   # 損失の計算
            loss.backward()                     # 逆方向計算(勾配計算)
            optimizer.step()                    # パラメータの更新

            # 履歴の累積
            train_loss += loss.item()
            train_batches += 1

        # validation_lossの計算
        model.eval()    # 推論モード
        with torch.no_grad():
            for i, data in enumerate(val_loader):   # バッチ毎に読み込む
                inputs, labels = data[0].to(device), data[1].to(device) # data は [inputs, labels] のリスト
                outputs = model(inputs)               # 順方向計算
                loss = criterion(outputs, labels)   # 損失の計算

                # 履歴の累積
                val_loss += loss.item()
                val_batches += 1

        # 履歴の出力
        print('epoch %d train_loss: %.10f' %
              (epoch + 1,  train_loss/train_batches))
        print('epoch %d val_loss: %.10f' %
              (epoch + 1,  val_loss/val_batches))

        with open("history.csv",'a') as f:
            print(str(epoch+1) + ',' + str(train_loss/train_batches) + ',' + str(val_loss/val_batches),file=f)

        # 最良モデルの保存
        if min_loss > val_loss/val_batches:
            min_loss = val_loss/val_batches
            PATH = "best.pth"
            torch.save(model.state_dict(), PATH)

        # 学習率の動的変更
        scheduler.step(val_loss/val_batches)

    # 最終エポックのモデル保存
    print("training finished")
    PATH = "lastepoch.pth"
    torch.save(model.state_dict(), PATH)


if __name__ == "__main__":
    train()
