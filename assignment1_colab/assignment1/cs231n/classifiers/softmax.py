from builtins import range
import numpy as np
from random import shuffle
try:
    from past.builtins import xrange
except ImportError:
    xrange = range


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_train = X.shape[0]
    for i in range(num_train):
        # 计算每个样本的得分，减去最大得分以提高数值稳定性
        scores = X[i].dot(W) # 1*C
        scores -= np.max(scores)
        p = np.exp(scores) / np.sum(np.exp(scores)) # 1*C,softmax概率分布
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss

        for class_idx in range(W.shape[1]):
            grad_contrib = p[class_idx] * X[i]
            if class_idx == y[i]:
                grad_contrib = (p[class_idx] - 1) * X[i]
                # 假如是正确的类别则对应另一个梯度公式
            dW[:, class_idx] += grad_contrib


    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2.0 * reg * W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    scores = X.dot(W) #直接进行矩阵乘法，不使用循环进行计算
    scores -= np.max(scores,axis=1,keepdims=True)

    exp = np.exp(scores)
    p = exp / np.sum(exp,axis=1,keepdims=True)
    logp = np.log(p)
    # 在logp中使用np.arange和y配对，直接取出对应正确类别的概率，然后求和得到总的损失，最后除以样本数得到平均损失
    loss = -np.sum(logp[np.arange(num_train),y]) / num_train + reg * np.sum(W * W)

    # 计算梯度，直接在p中修改正确类别的概率，然后使用高效的矩阵乘法
    dscores = p
    dscores[np.arange(num_train),y] -= 1 # 按索引操作
    dW = X.T.dot(dscores) / num_train + 2.0 * reg * W

    return loss, dW
