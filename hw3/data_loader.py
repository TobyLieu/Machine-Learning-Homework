import pandas as pd
import numpy as np


def dataLoader(path):
    train_df = pd.read_csv(path + "mnist_train.csv", sep=',')
    x_train = train_df.values[:, 1:] / 255
    y_train = train_df.values[:, 0]
    test_df = pd.read_csv(path + "mnist_test.csv", sep=',')
    x_test = test_df.values[:, 1:] / 255
    y_test = test_df.values[:, 0]
    return x_train, y_train, x_test, y_test


# x_train, y_train, x_test, y_test = dataLoader('~/Data/hw3_data/')
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
