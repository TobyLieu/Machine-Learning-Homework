from sklearn.cluster import KMeans
from data_loader import dataLoader
from cluster_measure import cluster_acc

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = dataLoader('~/Data/hw3_data/')
    estimator = KMeans(n_clusters=10, )
    estimator.fit(x_train)
    acc, _ = cluster_acc(estimator.labels_, y_train)
    print("kmeans acc is :", acc)