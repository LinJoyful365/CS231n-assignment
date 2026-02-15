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
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss

        for class_idx in range(W.shape[1]):
            grad_contrib = p[class_idx] * X[i]
            if class_idx == y[i]:
                grad_contrib -= X[i]
            dW[:, class_idx] += grad_contrib


    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

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


    score_matrix = X.dot(W)
    score_matrix -= np.max(score_matrix, axis=1, keepdims=True)  # 稳定性优先
    exp_scores = np.exp(score_matrix)
    prob_matrix = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    num_train = X.shape[0]
    correct_log_probs = -np.log(prob_matrix[np.arange(num_train), y])
    loss = np.mean(correct_log_probs) + reg * np.sum(W * W)

    prob_matrix[np.arange(num_train), y] -= 1.0
    dW = X.T.dot(prob_matrix) / num_train + 2.0 * reg * W
    return loss, dW
