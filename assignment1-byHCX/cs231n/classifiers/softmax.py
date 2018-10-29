import numpy as np
from random import shuffle

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
  - gradient with respect to weights ; an array of same shape as W
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
  num = X.shape[0]
  for i in range(num):
        score=X[i].dot(W)
        score -= np.max(score) #prevents numerical instability  分子分母同除以一个数，比值不变，防止求指数值过大
        score=np.exp(score)
        sum_s = np.sum(score)
        scoref = score/sum_s
        loss += -np.log(scoref[y[i]])
        for j in range(W.shape[1]):
            if j ==y[i]:
                dW[:,j]+=(score[j]/sum_s - 1)*X[i]
            else:
                dW[:,j]+=(score[j]/sum_s)*X[i]
  loss /= num
  loss += reg*np.sum(W*W)
  dW = dW/num + 2*reg*W
   
        
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  score=X.dot(W)
  score -= np.max(score) #prevents numerical instability  分子分母同除以一个数，比值不变，防止求指数值过大
  score=np.exp(score)
  sum_s = np.sum(score,axis=1)
  scoref = score/sum_s.repeat(W.shape[1]).reshape(score.shape)
  p_y=scoref[range(X.shape[0]),y]
  loss=np.sum(-np.log(p_y))
  scoref[range(X.shape[0]),y] -=1
  dW = np.dot(X.T, scoref)
  dW = dW/X.shape[0] + 2*reg*W
  loss = loss/X.shape[0] + reg*np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

