import numpy as np
import random
from data_loader import dataLoader
from cluster_measure import cluster_acc


def distance(x, y):
    """
    欧式距离: 矩阵x的第i行和矩阵y的第j行求距离, 存在z的(i, j)位置
    """
    z = np.expand_dims(x, axis=1) - y
    z = np.square(z)
    z = np.sqrt(np.sum(z, axis=2))
    return z


class myKmeans(object):
    def __init__(self, n_clusters, max_iter, gt, init=""):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.gt = gt
        self.center = 0
        self.cluster = 0

    def fit(self, data, label):
        # 初始化聚类中心
        indices = random.sample(range(data.shape[0]), self.n_clusters)
        self.center = np.copy(data[indices])

        # 初始化类别标记
        self.cluster = np.zeros(data.shape[0], dtype=np.int32)

        # 循环
        for i in range(self.max_iter):
            # 计算距离
            dis = distance(data, self.center)

            # 根据初试化得聚类中心划分类
            self.cluster = np.argmin(dis, axis=1)
            onehot = np.zeros(data.shape[0] * self.n_clusters, dtype=np.float32)
            onehot[self.cluster + np.arange(data.shape[0]) * self.n_clusters] = 1.
            onehot = np.reshape(onehot, (data.shape[0], self.n_clusters))

            # 计算新的聚类中心
            new_center = np.matmul(np.transpose(onehot, (1, 0)), data)
            new_center = new_center / np.expand_dims(np.sum(onehot, axis=0), axis=1)

            # 判断是否收敛
            if (np.linalg.norm(new_center - self.center) < 1e-3):
                break 
            self.center = new_center

            # 计算acc
            if (i % 10 == 0):
                acc, _ = cluster_acc(self.cluster, label)
                print('iter = %d, acc = %.4f' % (i, acc))


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = dataLoader('~/Data/hw3_data/')
    model = myKmeans(10, 200, y_train)
    model.fit(x_train, y_train)
    acc, _ = cluster_acc(model.cluster, y_train)
    print("my_kmeans acc is :", acc)
