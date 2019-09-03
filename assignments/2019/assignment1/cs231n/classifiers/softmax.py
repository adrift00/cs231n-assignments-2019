import numpy as np


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

    #############################################################################
    # Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]
    C = W.shape[1]
    for i in range(N):
        scores = X[i].dot(W)
        # 直接使用会过大，进行标准化
        stable_scores = scores - np.max(scores)
        stable_scores = np.exp(stable_scores)
        correct_scores = stable_scores[y[i]]
        loss_i = -np.log(correct_scores / np.sum(stable_scores))
        loss += loss_i
        # 计算梯度
        dScores = np.zeros(scores.shape)
        dScores = stable_scores / np.sum(stable_scores)
        dScores[y[i]] -= 1
        dW += X[i][:, np.newaxis].dot(dScores[np.newaxis, :])  # 两个一维向量相乘成一个矩阵，所以这么写

    loss = loss / N + reg * np.sum(W * W)
    dW = dW / N + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]
    C = W.shape[1]
    scores = X.dot(W)
    stable_scores = scores - np.max(scores, axis=1, keepdims=True)
    stable_scores = np.exp(stable_scores)
    loss = np.sum(
        -np.log(stable_scores[np.arange(N), y] / np.sum(stable_scores, axis=1)))  # 这里np.sum不能keep_dims,否则广播会出错
    loss = loss / N + reg * np.sum(W * W)
    # 计算梯度
    dScores = stable_scores / np.sum(stable_scores, axis=1, keepdims=True)
    dScores[np.arange(N), y] -= 1
    dScores /= N
    dW = X.T.dot(dScores)
    dW = dW + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
