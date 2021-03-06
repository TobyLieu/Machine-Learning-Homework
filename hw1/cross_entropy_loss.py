import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    if(s == 1):
        return s - 1e-5
    return s


train_frame = pd.read_csv('./Data/mnist_01_train.csv').values
test_frame = pd.read_csv('./Data/mnist_01_test.csv').values
train_data = train_frame[:, 1:] / 255
train_label = train_frame[:, 0]
test_data = test_frame[:, 1:] / 255
test_label = test_frame[:, 0]

epoch = 100
lr = 2
w = np.random.random((train_data.shape[1], 1))
n = train_data.shape[0]
dim = train_data.shape[1]
x = train_data
y = train_label

cel_old_loss = 100
loss_list = []
loss_x = []
for i in range(epoch):
    cel_grad_sum = np.zeros([784, 1])
    cel_loss_sum = 0
    for l in range(n):
        cel_grad_sum += (sigmoid(np.dot(x[l], w)) -
                         y[l]) * x[l].reshape(784, 1)
        cel_loss_sum += -(y[l] * np.log(sigmoid(np.dot(x[l], w))) +
                          (1 - y[l]) * np.log(1 - sigmoid(np.dot(x[l], w))))
        # print(cel_loss_sum)
    cel_grad = cel_grad_sum / n
    cel_loss = cel_loss_sum / n

    loss_list.append(cel_loss[0][0])
    loss_x.append(i)

    print(i, cel_loss)
    if(abs(cel_old_loss - cel_loss) < 1e-6):
        break
    w = w - lr * cel_grad
    cel_old_loss = cel_loss

right = 0
for l in range(test_data.shape[0]):
    if(abs(sigmoid(np.dot(test_data[l], w)) - test_label[l]) <= 0.5):
        right += 1

cel_acc = right / test_data.shape[0]
print("cel_acc =", cel_acc)

plt.plot(loss_x, loss_list)
# plt.savefig('./Figure/' + str(epoch) + '+' + str(lr) + '.png')
# plt.show()
