import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

# hyper parameter
criterion = nn.CrossEntropyLoss()
batch_size = 64
num_inputs = 3072
num_outputs = 10
learning_rate = 1e-2
momentum = 0.8
beta1 = 0.99
num_epochs = 50

hidden_layer_size = 5
layer_num = 2
kenel_num = 6
net_type = 'resnet'

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在把图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),  #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# cifar10路径
cifar10Path = '~/Data/cifar'

# 训练数据集
train_dataset = torchvision.datasets.CIFAR10(root=cifar10Path,
                                             train=True,
                                             transform=transform_train,
                                             download=True)
# 测试数据集
test_dataset = torchvision.datasets.CIFAR10(root=cifar10Path,
                                            train=False,
                                            transform=transform_test)

# 训练数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
# 测试数据加载器
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = x.view(-1, num_inputs)
        return F.softmax(self.linear(x), dim=1)


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()  #
        self.fc1 = torch.nn.Linear(num_inputs, hidden_layer_size)  # 第一个隐藏层
        # self.fc2 = torch.nn.Linear(hidden_layer_size, hidden_layer_size)  # 第二个隐藏层
        # self.fc3 = torch.nn.Linear(hidden_layer_size, hidden_layer_size)  # 第三个隐藏层
        self.fcout = torch.nn.Linear(hidden_layer_size, num_outputs)  # 输出层

    def forward(self, x):
        x = x.view(-1, num_inputs)  # 将一个多行的Tensor,拼接成一行
        x = F.relu(self.fc1(x))  # 使用 relu 激活函数
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = F.softmax(self.fcout(x), dim=1)  # 输出层使用 softmax 激活函数
        return x


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2, 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel,
                      outchannel,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False), nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,
                      outchannel,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.BatchNorm2d(outchannel))
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel,
                          outchannel,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(outchannel))

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():

    return ResNet(ResidualBlock)


# model = LinearNet()
# model = MLPNet()
model = CNNNet()
# model = ResNet18()
optimizer = optim.SGD(params=model.parameters(),
                      lr=learning_rate,
                      momentum=momentum)
'''optimizer = optim.Adam(params=model.parameters(),
                       lr=learning_rate,
                       betas=(beta1, 0.999))'''
# optimizer = optim.SGD(params=model.parameters(), lr=learning_rate)

if (torch.cuda.is_available()):
    model = model.cuda()
    criterion = criterion.cuda()


def train(epoch):
    # 每次输入barch_idx个数据
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if (torch.cuda.is_available()):
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        # loss
        loss = criterion(output, target)
        loss.backward()
        # update
        optimizer.step()
        if batch_idx % 100 == 0:
            if (torch.cuda.is_available()):
                data, loss = data.cpu(), loss.cpu()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))


x = []
acc_list = []


def test(epoch):
    with torch.no_grad():
        test_loss = 0
        correct = 0
        # 测试集
        for data, target in test_loader:
            if (torch.cuda.is_available()):
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += criterion(output, target).data.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)

        x.append(epoch)
        acc_list.append(100. * correct / len(test_loader.dataset))

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
              format(test_loss, correct, len(test_loader.dataset),
                     100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    for epoch in range(1, num_epochs):
        if (epoch == 20):
            learning_rate = 1e-3
        train(epoch)
        test(epoch)

    x = np.array(x)
    acc_list = np.array(acc_list)
    np.savez('./result/' + net_type + '/lr_alter, batch_size=128', x, acc_list)
