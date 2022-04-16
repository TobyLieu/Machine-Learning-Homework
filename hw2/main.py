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
momentum = 0.5
num_epochs = 50

hidden_layer_size = 5
layer_num = 2
kenel_num = 6
net_type = 'cnn'

# 数据预处理
transform = transforms.Compose([
    # +4填充至36x36
    transforms.Pad(4),
    # 随机水平翻转
    transforms.RandomHorizontalFlip(),
    # 随机裁剪至32x32
    transforms.RandomCrop(32),
    # 转换至Tensor
    transforms.ToTensor(),
    # 归一化
    transforms.Normalize(std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5))
])

# cifar10路径
cifar10Path = '~/Data/cifar'

#  训练数据集
train_dataset = torchvision.datasets.CIFAR10(root=cifar10Path,
                                             train=True,
                                             transform=transform,
                                             download=True)
# 测试数据集
test_dataset = torchvision.datasets.CIFAR10(root=cifar10Path,
                                            train=False,
                                            transform=transform)

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


# model = LinearNet(num_inputs, num_outputs)
# model = MLPNet(num_inputs, num_outputs)
model = CNNNet()
optimizer = optim.SGD(params=model.parameters(),
                      lr=learning_rate,
                      momentum=momentum)

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
        train(epoch)
        test(epoch)

    x = np.array(x)
    acc_list = np.array(acc_list)
    np.savez('./result/' + net_type + '/momentum=' + momentum, x, acc_list)
