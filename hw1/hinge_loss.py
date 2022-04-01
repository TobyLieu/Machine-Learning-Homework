import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


train_frame = pd.read_csv('./Data/mnist_01_train.csv').values
test_frame = pd.read_csv('./Data/mnist_01_test.csv').values
train_data = train_frame[:, 1:] / 255
train_label = train_frame[:, 0]
test_data = test_frame[:, 1:] / 255
test_label = test_frame[:, 0]

for i in range(train_label.shape[0]):
    if(train_label[i] == 0):
        train_label[i] = -1
for i in range(test_label.shape[0]):
    if(test_label[i] == 0):
        test_label[i] = -1

epoch = 1000
lr = 1
w = np.random.randn(train_data.shape[1], 1)
n = train_data.shape[0]
dim = train_data.shape[1]
x = train_data
y = train_label

hin_old_loss = 100
for i in range(epoch):
    hin_grad_sum = np.zeros([784, 1])
    hin_loss_sum = 0
    for l in range(n):
        y_tmp = np.dot(x[l].reshape(1, 784), w)
        if(y_tmp * y[l] < 1):
            hin_grad_sum += -y[l] * x[l].reshape(784, 1)
            hin_loss_sum += 1 - y_tmp * y[l]
        # print(hin_loss_sum)
    hin_grad = hin_grad_sum / n
    hin_loss = hin_loss_sum / n
    print(i, hin_loss)
    if(abs(hin_old_loss - hin_loss) < 1e-5):
        break
    w = w - lr * hin_grad
    hin_old_loss = hin_loss

right = 0
for l in range(test_data.shape[0]):
    if(np.dot(test_data[l], w) * test_label[l] > 0):
        right += 1

hin_acc = right / test_data.shape[0]
print("hin_acc =", hin_acc)
