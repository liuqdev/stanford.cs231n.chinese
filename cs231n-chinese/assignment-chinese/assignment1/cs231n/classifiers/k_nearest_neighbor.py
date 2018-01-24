# -*- coding: utf-8 -*-
import numpy as np

class KNearestNeighbor(object):
  """ 使用 L2 距离的 kNN 分类器 """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    训练分类器,对于kNN来说只是对训练数据简单记忆.

    Inputs:
    - X: 一个 numpy array, 大小为 (num_train, D) 表示训练集中有 num_train个训练数据.
        每个训练数据的维数为D.
    - y: 一个 numpy array, 大小为 (N,) 包含训练的标签, 其中
         y[i] 是 X[i] 的标签.
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
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    使用在训练数据和测试数据上的嵌套循环, 计算 X 中每个测试点与训练集 self.X_train
    上每个点之间的距离.
    
    输入:
    - X: 一个 numpy array, 大小为 (num_test, D), 包含测试数据.

    返回:
    - dists: 一个 numpy array, 大小为 (num_test, num_train) 其中 dists[i, j]
      表示测试集中第 i 个点和训练集中第 j 个点之间的欧式距离.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    for i in xrange(num_test):
      for j in xrange(num_train):
        #####################################################################
        # 待完善:                                        #
        # 计算出第 i 个 测试点(D 维) 和 第 j 个训练点之间的 L2 距离    #
        # 并将结果存储在 dists[i, j] 中. 对于纬度上的操作不要使用循环   #
        #####################################################################
        # 通过
        dists[i][j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))
        #####################################################################
        #                       代码结束                          #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    通过在测试集数据上遍历一次, 来计算出测试集上 X 和训练集 self.X_train 上每个数据
    间的距离

    输入 / 输出: 和 compute_distances_two_loops 一样
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      #######################################################################
      # 待完善:                                                     #
      # 计算出第 i 个测试点和所有训练点的 L2 距离 #
      # 将其存储于 dists[i, :].                        #
      #######################################################################
      # 通过
      dists[i] = np.sqrt(np.sum(np.square(self.X_train - X[i]), axis = 1))
      #######################################################################
      #                         代码结束                           #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    不含显式的循环,来计算出测试集上 X 和训练集 self.X_train 上每个数据
    间的距离
 
    输入 / 输出: 和 compute_distances_two_loops 一致
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # 待完善:                                           #
    #  不含显式的循环,来计算出测试集上 X 和训练集 self.X_train 上每个数据   #
    #  间的 L2 距离 , 将结果存储于 dists 中      #
    #                                                           #
    #                                                           #
    # 你应当使用基本的 array 操作来实现本函数; #
    # 特别说明,不能使用 scipy 提供的内置函数.                #
    #                                                           #
    # 提示: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                     #
    #########################################################################
    # 通过
    dists = np.sqrt(-2*np.dot(X, self.X_train.T) + np.sum(np.square(self.X_train), axis = 1) + np.transpose([np.sum(np.square(X), axis = 1)]))
    #########################################################################
    #                         代码结束                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
   给定一个距离矩阵, 包含测试集和训练集之间点的距离.
   预测每个测试集的标签.

    输入:
    - dists: 一个 numpy array 大小(num_test, num_train) 其中 dists[i, j]
      为测试集中第 i 个点和训练集中第 j 个点的距离.

    返回:
    - y: 一个 numpy array 大小为(num_test,) 包含每个测试数据的标签.
      其中 y[i]是对测试点 X[i] 标签的预测.  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # 一个 list 长度为 k ,用来保存第 i 个测试点,与之相邻最近的 k 个点

      closest_y = []
      #########################################################################
      # 待完善:                                                                 #
      # 根据距离矩阵来获得距离第 i 个测试点最近的 k 个点   #
      # 使用 self.y_train 找到这些邻近点所属标签       #
      # 将标签值存储于 closest_y.                           #
      # 提示: 查找函数使用 numpy.argsort.                             #
      #########################################################################
      # 通过      
      closest_y = self.y_train[np.argsort(dists[i])[:k]]
      #########################################################################
      # 待完善:                                                                 #
      # 现在我们获取了最近的 k 个邻居点,     #
      # 接下来需要找到 cloest_y 中最常见的标签   #
      # 将该标签存储于 y_pred[i] 中. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      # 通过
      y_pred[i] = np.argmax(np.bincount(closest_y))
      #########################################################################
      #                          代码结束                           # 
      #########################################################################

    return y_pred

