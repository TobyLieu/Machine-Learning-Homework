import numpy as np
from sklearn.datasets import load_iris
import clustering as cl
from sklearn import mixture


def multi_norm(x, mu, sigma):
    """
    return the probability of Multidimensional gaussian distribution, and there is a better implementation in scipy.stats
    返回多维高斯分布的结果, 该方法在scipi库中有更好的实现。
    :param x: x
    :param mu: mean vector
    :param sigma: covariance matrix
    :return: the probability of Multidimensional gaussian distribution
    """
    det = np.linalg.det(sigma)
    inv = np.matrix(np.linalg.inv(sigma))
    x_mu = np.matrix(x - mu).T
    const = 1 / (((2 * np.pi)**(len(x) / 2)) * (det**(1 / 2)))
    exp = -0.5 * x_mu.T * inv * x_mu
    return float(const * np.exp(exp))


def distance(a, b, p):
    a = np.array(a)
    b = np.array(b)
    return np.sum((a - b)**p)**(1 / p)


def pairing(data, truth, label):
    datatemp = data.copy()
    centerTruth = []
    center = []
    new_label = np.zeros(label.shape) - 1
    for i in range(0, np.max(truth) + 1):
        centerTruth.append(
            list(np.mean(datatemp[np.argwhere(truth == i)], axis=0)))
        center.append(list(np.mean(datatemp[np.argwhere(label == i)], axis=0)))
    for i in range(0, np.max(truth) + 1):
        temp = []
        for j in range(0, np.max(truth) + 1):
            temp.append(distance(centerTruth[i], center[j], 2))
        number = temp.index(min(temp))
        print(number)
        new_label[label == number] = i
    return new_label


def calcu_acc(truth, label):
    temp = np.zeros(label.shape)
    temp[np.argwhere(label == truth)] = 1
    return np.sum(temp) / label.shape[0]


class GMM(object):
    def __init__(self, n_clusters, max_iter, init_params="random"):
        """
        初始化变量：聚类数目以及最大迭代数
        :param n_clusters: 最大迭代数
        """
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.init_params = init_params
        self.num = None
        self.dim = None
        self.X = None
        self.Q = None
        self.weight = None
        self.covar = None
        self.mu = None
        self.labels = None

    def _initialize_params(self, X):
        """
        初试化模型参数,
        :param X: 分类的数据集
        :return:
        """
        self.X = X  # 分类的数据集
        self.num = X.shape[0]  # 样本数目
        self.dim = X.shape[1]  # 特征维度
        self.Q = np.zeros((self.num, self.n_clusters))  # 初始化各高斯分布对观测数据的响应度矩阵
        if self.init_params == "random":
            self.weight = [1 / self.n_clusters
                           ] * self.n_clusters  # 初始化各高斯分布的权重为聚类数目分之一
            self.mu = np.random.uniform(0, 1,
                                        (self.n_clusters, self.dim)) * np.max(
                                            X, axis=0)  # 随机产生均值向量
            self.covar = np.array([
                np.identity(self.dim) for _ in range(self.n_clusters)
            ])  # 随机产生协方差矩阵
        if self.init_params == "kmeans":
            kmeanmodel = cl.Mykmean(self.n_clusters, 20)
            kmeanmodel.fit(self.X)
            self.mu = kmeanmodel.center
            self.weight = []
            # print(kmeanmodel.labels)
            for i in range(self.n_clusters):
                temp = np.zeros(kmeanmodel.labels.shape)
                temp[kmeanmodel.labels == i] = 1
                self.weight.append(np.sum(temp) / self.num)
            self.covar = np.zeros((self.n_clusters, self.dim, self.dim))
            for i in range(self.n_clusters):
                temp = self.X.copy()
                # print(np.argwhere(kmeanmodel.labels == i).T.tolist())
                temp = temp[np.argwhere(
                    kmeanmodel.labels == i).T.tolist()[0], :]
                self.covar[i, :, :] = np.cov(temp, rowvar=False)
            # print(self.mu)
            # print(self.covar)
            # print(self.weight)

    def e_step(self):
        """
        e步, 更新分模型对数据的响应度矩阵Q, 计算公式为
        :return:
        """
        for i in range(self.num):
            q_i = []
            for k in range(0, self.n_clusters):
                postProb = multi_norm(self.X[i, :], self.mu[k, :],
                                      self.covar[k, :, :])
                q_i.append(self.weight[k] * postProb)
            self.Q[i, :] = np.array(q_i) / np.sum(q_i)

    def m_step(self):

        # update weight 更新权值矩阵
        self.weight = np.mean(self.Q, axis=0)

        # update mu 更新均值向量
        temp = []
        for k in range(self.n_clusters):
            up = np.zeros(self.dim)
            for j in range(self.num):
                up += self.Q[j, k] * np.array(self.X[j, :])
            down = np.sum(self.Q[:, k])
            temp.append(up / down)
        self.mu = np.array(temp)

        # update covar
        for k in range(self.n_clusters):
            up = np.zeros((self.dim, self.dim))
            for j in range(self.num):
                x_mu = np.matrix(self.X[j, :] - self.mu[k, :])
                # print(x_mu.T*x_mu)
                up += self.Q[j, k] * (x_mu.T * x_mu)
            # print(up)
            down = np.sum(self.Q[:, k])
            var = np.array(up / down)
            self.covar[k, :, :] = var

    def fit(self, X):
        self.X = X
        self._initialize_params(X)
        while self.max_iter > 0:
            # 初始化变量
            # e-step
            self.e_step()
            # m-step
            self.m_step()
            self.max_iter -= 1
        self.labels = np.argmax(self.Q, axis=1)


if __name__ == '__main__':
    iris = load_iris()
    # # 测试post_prob
    # from scipy.stats import multivariate_normal
    # mean = [0, 0,2]
    # cov = [[1, 0,0], [0, 1,0],[0,0,1]]
    # x = [18, 2, 1]
    # var = multivariate_normal(mean, cov)
    # print(var.pdf([18, 2,1]))
    # print(multi_norm(np.array(x), np.array(mean), np.array(cov)))

    # 鸢尾花的数据对比
    gmm = mixture.GaussianMixture(n_components=3,
                                  init_params="kmeans").fit(iris.data)
    labels = gmm.predict(iris.data)
    label = pairing(iris.data, iris.target, labels)
    acc = calcu_acc(iris.target, label)
    print("GMM acc is :", acc)

    mygmmmodel = GMM(3, 100, init_params="kmeans")
    mygmmmodel.fit(iris.data, )
    label = pairing(iris.data, iris.target, mygmmmodel.labels)
    acc = calcu_acc(iris.target, label)
    print("MYGMM acc is :", acc)