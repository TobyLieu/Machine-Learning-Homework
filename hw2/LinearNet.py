import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data_loader import load_data
from torch.autograd import Variable

X_train, Y_train, X_test, Y_test = load_data('../../Data/ml_data_hw2')
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
Y_train = torch.from_numpy(np.array(Y_train)).long()
Y_test = torch.from_numpy(np.array(Y_test)).long()

train_num = X_train.shape[0]
test_num = X_test.shape[0]

batch_size = 64

num_inputs = 3072
num_outputs = 10


class LinearNet(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):  # x shape: (batch, 1, 28, 28)
        x = x.view(-1, num_inputs)
        return F.softmax(self.linear(x), dim=1)


net = LinearNet(num_inputs, num_outputs)
optimizer = optim.SGD(params=net.parameters(), lr=0.1)


def train(epoch):
    # 每次输入barch_idx个数据
    batch_idx = 0
    while (True):
        down = batch_idx * batch_size
        up = (batch_idx + 1) * batch_size if (
            batch_idx + 1) * batch_size <= train_num else train_num
        X = Variable(X_train[down:up, :])
        Y = Variable(Y_train[down:up])

        optimizer.zero_grad()
        output = net(X)
        # loss
        loss = F.cross_entropy(output, Y)
        loss = loss.requires_grad_()
        loss.backward()
        # update
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, train_num,
                100. * batch_idx * batch_size / train_num, loss.data.item()))
            for parameters in net.parameters():
                print(parameters)

        if (up == train_num):
            break

        batch_idx += 1


def test():
    test_loss = 0
    correct = 0
    # 测试集
    for i in range(test_num):
        X = X_test[i, :]
        Y = Y_test[i].reshape(1)

        output = net(X)
        # sum up batch loss
        test_loss += F.cross_entropy(output, Y).data.item()
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        if (pred == Y):
            correct += 1

    test_loss /= test_num
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, test_num, 100. * correct / test_num))


for epoch in range(1, 100):
    train(epoch)
    test()