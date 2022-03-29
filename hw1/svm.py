import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC

train_frame = pd.read_csv('./Data/mnist_01_train.csv').values
test_frame = pd.read_csv('./Data/mnist_01_test.csv').values
train_data = train_frame[:, 1:]
train_label = train_frame[:, 0]
test_data = test_frame[:, 1:]
test_label = test_frame[:, 0]

linear_svm = SVC(kernel="linear")
linear_svm.fit(train_data, train_label)
linear_acc = linear_svm.score(test_data, test_label)

rbf_svm = SVC(kernel="rbf")
rbf_svm.fit(train_data, train_label)
rbf_acc = rbf_svm.score(test_data, test_label)

print("linear accuracy =", linear_acc, ";", "rbf accuracy =", rbf_acc)