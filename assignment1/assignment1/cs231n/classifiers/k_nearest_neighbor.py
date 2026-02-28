from builtins import range
from builtins import object
import numpy as np
try:
    from past.builtins import xrange
except ImportError:
    xrange = range


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))# 直接创建一个距离阵列，后续填充
        for i in range(num_test):
            for j in range(num_train):
                delta_vec = X[i] - self.X_train[j]
                dists[i, j] = np.sqrt(delta_vec.dot(delta_vec))
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            delta_matrix = self.X_train - X[i]  # 训练数据矩阵减去测试数据的第i行，得到一个差值矩阵,也就是第i个测试数据与所有训练数据的差值
            dists[i,:] = np.sqrt(np.sum(delta_matrix * delta_matrix,axis=1)) #逐元素相乘（平方），然后再按列求和，最后开方得到距离
        return dists
        


    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        #在真实的计算情况下，拓宽到3D矩阵会导致内存不足，计算缓慢
        # num_test = X.shape[0]
        # num_train = self.X_train.shape[0]
        # dists = np.zeros((num_test, num_train))
        # #这里我们使用numpy的广播机制来计算
        # X_train_shape = self.X_train.reshape(1, num_train, -1) #将训练数据重塑为（1，num_train，D）的形状
        # X_shape = X.reshape(num_test, 1, -1) #将测试数据重塑为（num_test，1，D）的形状
        # delta_matrix = X_train_shape - X_shape #利用广播机制计算差值矩阵，得到一个（num_test，num_train，D）的矩阵
        # dists = np.sqrt(np.sum(delta_matrix * delta_matrix,axis=2)) #在最后一个维度上求和，得到一个（num_test，num_train）的矩阵
        # return dists

        #这里使用公式||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b
        test_sum_square = np.sum(X * X, axis=1) #每行的平方和，得到一个（num_test，）的向量
        train_sum_square = np.sum(self.X_train * self.X_train, axis=1) #得到一个（num_train,)的向量        
        inner_product = X.dot(self.X_train.T)
        square_dists = test_sum_square[:,np.newaxis] + train_sum_square.reshape(1, -1) - 2 * inner_product #根据公式计算平方距离
        dists = np.sqrt(square_dists) #开方得到距离
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            if i < 3:
                idx = np.argsort(dists[i])[0]
                print(f"Test {i} closest train index: {idx}, pred label: {self.y_train[idx]}")
            closest_y = []
            closest_y = self.y_train[np.argsort(dists[i])[:k]] #对第i行(第i个测试数据)的距离进行排序，取前k个索引，然后用这些索引从训练标签中取出对应的标签
            vote_count = np.bincount(closest_y.astype(np.int64))#统计closest_y中每个标签的出现次数，返回一个数组，其中第i个元素表示标签i出现的次数
            y_pred[i] = np.argmax(vote_count) #统计每个标签的出现次数，取出现次数最多的标签作为预测结果，如果有多个标签出现次数相同，np.argmax会返回第一个出现的最大值的索引，也就是较小的标签
        return y_pred