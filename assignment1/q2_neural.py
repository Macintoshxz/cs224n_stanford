#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def cross_entropy(x, y):
    assert x.shape == y.shape
    # assume that predictions and labels are the columns of x and y
    # assume that either x or y is one hot - so log is applied after
    #   dot product and this operation becomes communative
    return [np.log(x[:, ci].T.dot(y[:, ci])) for ci in range(0, x.shape[1])]


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    # W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    # ofs += Dx * H
    # b1 = np.reshape(params[ofs:ofs + H], (1, H))
    # ofs += H
    # W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    # ofs += H * Dy
    # b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    W1 = np.reshape(params[ofs:ofs + Dx * H], (H, Dx))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (H, 1))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (Dy, H))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (Dy, 1))
    data = data.T # now each column is a training example
    labels = labels.T # now each column is label vector


    ### YOUR CODE HERE: forward propagation
    a1 = data
    z2 = W1.dot(a1) + b1
    a2 = sigmoid(z2)
    z3 = W2.dot(a2) + b2
    a3 = softmax(z3)
    cost = cross_entropy(labels, a3)
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    delta3 = np.identity().dot()
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
