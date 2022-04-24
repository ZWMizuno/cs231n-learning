from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    num_dimension = X.shape[1]
    scores = X.dot(W)
    exp_scores = np.exp(scores)
    pro = exp_scores/np.reshape(np.sum(exp_scores,axis=1),(num_train,1))#不reshape则报错operands could not be broadcast together with shapes (500,10) (500,)
    loss_i = -np.log(pro[np.arange(num_train),y]/1)
    loss += np.sum(loss_i)/num_train
    loss += reg * np.sum(W * W)

    sum_exp = np.sum(exp_scores, axis=1)
    coe = exp_scores / np.reshape(sum_exp,(exp_scores.shape[0],1)) #sum之后要reshape
    coe[np.arange(coe.shape[0]),y] -= 1
    sum_li = X.T.dot(coe)
    dW = sum_li / num_train + 2 * reg * W

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
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    num_dimension = X.shape[1]
    scores = X.dot(W)
    exp_scores = np.exp(scores)
    pro = exp_scores/np.reshape(np.sum(exp_scores,axis=1),(num_train,1))#不reshape则报错operands could not be broadcast together with shapes (500,10) (500,)
    loss_i = -np.log(pro[np.arange(num_train),y]/1)
    loss += np.sum(loss_i)/num_train
    loss += reg * np.sum(W * W)

    sum_exp = np.sum(exp_scores, axis=1)
    coe = exp_scores / np.reshape(sum_exp, (exp_scores.shape[0], 1))# sum之后要reshape
    coe[np.arange(coe.shape[0]),y] -= 1
    sum_li = X.T.dot(coe)
    dW = sum_li / num_train + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
