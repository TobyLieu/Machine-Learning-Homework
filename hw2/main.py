import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable

# hyper parameter
criterion = F.cross_entropy
batch_size = 64
num_inputs = 3072
num_outputs = 10
learning_rate = 1e-2
momentum = 0.9
num_epochs = 5

# 数据增广方法
transform = transforms.Compose([
    # +4填充至36x36
    transforms.Pad(4),
    # 随机水平翻转
    transforms.RandomHorizontalFlip(),
    # 随机裁剪至32x32
    transforms.RandomCrop(32),
    # 转换至Tensor
    transforms.ToTensor(),
])

# cifar10路径
cifar10Path = '../../Data/cifar'

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

    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):  # x shape: (batch, 1, 28, 28)
        x = x.view(-1, num_inputs)
        return F.softmax(self.linear(x), dim=1)


hidden_layer_size_1 = 4000


class MLPNet(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(MLPNet, self).__init__()  #
        self.fc1 = torch.nn.Linear(num_inputs, hidden_layer_size_1)  # 第一个隐含层
        # self.fc2 = torch.nn.Linear(1024, 128)  # 第二个隐含层
        self.fcout = torch.nn.Linear(hidden_layer_size_1, num_outputs)  # 输出层

    def forward(self, x):
        x = x.view(-1, num_inputs)  # 将一个多行的Tensor,拼接成一行
        x = F.relu(self.fc1(x))  # 使用 relu 激活函数
        # x = F.relu(self.fc2(x))
        x = F.softmax(self.fcout(x), dim=1)  # 输出层使用 softmax 激活函数
        return x


class CNNNet(nn.Module):

    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 最后是一个十分类，所以最后的一个全连接层的神经元个数为10

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)  # 展平  x.size()[0]是batch size
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


def train(epoch):
    # 每次输入barch_idx个数据
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        # loss
        loss = criterion(output, target)
        loss.backward()
        # update
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))


def test():
    with torch.no_grad():
        test_loss = 0
        correct = 0
        # 测试集
        for data, target in test_loader:
            # data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).data.item()
            # get the index of the max
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
              format(test_loss, correct, len(test_loader.dataset),
                     100. * correct / len(test_loader.dataset)))


for epoch in range(1, 20):
    train(epoch)
    test()