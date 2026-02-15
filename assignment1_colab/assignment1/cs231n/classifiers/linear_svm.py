from builtins import range
import numpy as np
from random import shuffle
try:
    from past.builtins import xrange
except ImportError:
    xrange = range


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).
    """
    dW = np.zeros(W.shape)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for sample_idx in range(num_train):
        score_vec = X[sample_idx].dot(W)
        correct_score = score_vec[y[sample_idx]]
        for class_idx in range(num_classes):
            if class_idx == y[sample_idx]:
                continue
            margin_val = score_vec[class_idx] - correct_score + 1.0
            if margin_val > 0:
                loss += margin_val
                dW[:, class_idx] += X[sample_idx]
                dW[:, y[sample_idx]] -= X[sample_idx]

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2.0 * reg * W
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.
    """
    num_train = X.shape[0]
    score_matrix = X.dot(W)
    correct_scores = score_matrix[np.arange(num_train), y][:, None]
    margin_matrix = score_matrix - correct_scores + 1.0
    margin_matrix[np.arange(num_train), y] = 0.0
    margin_matrix = np.maximum(0.0, margin_matrix)

    loss = np.sum(margin_matrix) / num_train + reg * np.sum(W * W)

    positive_mask = (margin_matrix > 0).astype(np.float64)
    positive_count = np.sum(positive_mask, axis=1)
    positive_mask[np.arange(num_train), y] = -positive_count
    dW = X.T.dot(positive_mask) / num_train + 2.0 * reg * W
    return loss, dW
