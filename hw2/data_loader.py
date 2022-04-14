import torch
import numpy as np


def load_data(dir):
    import pickle
    import numpy as np
    X_train = []
    Y_train = []
    for i in range(1, 6):
        with open(dir + r'/data_batch_' + str(i), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        X_train.append(dict[b'data'])
        Y_train += dict[b'labels']
    X_train = np.concatenate(X_train, axis=0)

    with open(dir + r'/test_batch', 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X_test = dict[b'data']
    Y_test = dict[b'labels']

    return X_train, Y_train, X_test, Y_test


'''X_train, Y_train, X_test, Y_test = load_data('./data')
print(X_train.shape)
print(X_test.shape)
print(len(Y_train))
print(len(Y_test))
print(type(X_train))
print(type(Y_test))

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
Y_train = torch.from_numpy(np.array(Y_train))
Y_test = torch.from_numpy(np.array(Y_test))

print(X_train.shape)
print(Y_train.shape)'''