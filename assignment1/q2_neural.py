#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def cross_entropy(x, y):
    assert x.shape == y.shape
    return (np.log(x).T).dot(y)


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
    # print("dimensions:", dimensions)
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

    data = data.T  # now each column is a training example
    labels = labels.T  # now each column is label vector

    # print('W1:', W1.shape)
    # print('b1:', b1.shape)
    # print('W2:', W2.shape)
    # print('b2:', b2.shape)

    # print('data:', data.shape)
    # print('labels:', labels.shape)

    ### YOUR CODE HERE: forward propagation
    a1 = data
    z2 = W1.dot(a1) + b1
    a2 = sigmoid(z2)
    z3 = W2.dot(a2) + b2
    a3 = softmax(z3)
    # sum cost function so can use matrix multiplication to compute gradients
    cost = np.sum(cross_entropy(labels, a3))
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    delta3 = labels - a3
    # print('d3:', delta3.shape)
    # sum the b2 gradient for the batch so can use matrix multiplication for 
    #   W2 and W1 grad
    gradb2 = np.sum(delta3, axis=1)
    # print('gb2', gradb2.shape)
    # print("a2 and delta3 shape:",a2.shape,delta3.shape)
    # print(delta3.dot(a2.T).shape)
    # gradW2batch = np.array([delta3[:, ci:ci + 1].dot(a2[:, ci:ci + 1].T)
    #                         for ci in range(0, delta3.shape[1])])
    # print('gW2batch', gradW2batch.shape)
    # gradW2 = np.mean(gradW2batch, axis=0)
    # print('gW2', gradW2.shape)
    gradW2 = delta3.dot(a2.T)
    delta2 = W2.T.dot(delta3) * sigmoid_grad(z2)
    # print('d2', delta2.shape)
    gradb1 = np.sum(delta2, axis=1)
    # print('gb1', gradb1.shape)
    gradW1 = delta2.dot(a1.T)
    # gradW1batch = np.array([delta2[:, ci:ci + 1].dot(a1[:, ci:ci + 1].T)
    #                         for ci in range(0, delta2.shape[1])])
    # print('gW1batch', gradW1batch.shape)
    # gradW1 = np.mean(gradW1batch, axis=0)
    # print('gW1', gradW1.shape)
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.T.flatten(), gradb1.T.flatten(),
                           gradW2.T.flatten(), gradb2.T.flatten()))

    # print(gradb2,gradW2,gradb1,gradW2)

    return (cost, grad)


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
    for i in range(N):
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
