#!/usr/bin/python3
# -*- coding: utf-8 -*-
from kmaxpooling import pool
import numpy as np


def test_sequence_k1():
    data = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                     [1, 2, 3, 4, 5, 6, 7, 8],
                     [1, 2, 3, 4, 5, 6, 7, 8]]).reshape((1, 3, 8, 1))
    k = (1, 1)
    argmax = np.zeros((1, 3, 4, 1))
    outputs = np.zeros((1, 3, 4, 1))
    pool(inputs=data, kernel=(1, 5), outputs=outputs, padding=0, strides=(1, 1), argmax=argmax, k=k)
    assert (outputs == np.array([[[[5], [6], [7], [8]],
                                  [[5], [6], [7], [8]],
                                  [[5], [6], [7], [8]]]])).all()
    assert (argmax == np.array([[[[4], [5], [6], [7]],
                                 [[12], [13], [14], [15]],
                                 [[20], [21], [22], [23]]]])).all()
def test_sequence_k3():
    data = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                     [1, 2, 3, 4, 5, 6, 7, 8],
                     [1, 2, 3, 4, 5, 6, 7, 8]]).reshape((1, 3, 8, 1))
    k = (1, 3)
    argmax = np.zeros((1, 3, 4, 1))
    outputs = np.zeros((1, 3, 4, 1))
    pool(inputs=data, kernel=(1, 5), outputs=outputs, padding=0, strides=(1, 1), argmax=argmax, k=k)
    assert (outputs == np.array([[[[5], [6], [7], [8]],
                                  [[5], [6], [7], [8]],
                                  [[5], [6], [7], [8]]]])).all()
    assert (argmax == np.array([[[[4], [5], [6], [7]],
                                 [[12], [13], [14], [15]],
                                 [[20], [21], [22], [23]]]])).all()

def test_sequence_k2_2():
    data = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                     [1, 2, 3, 4, 5, 6, 7, 8],
                     [1, 2, 3, 4, 5, 6, 7, 8]]).reshape((1, 3, 8, 1))
    k = (2, 2)
    argmax = np.zeros((1, 3, 4, 1))
    outputs = np.zeros((1, 3, 4, 1))
    pool(inputs=data, kernel=(1, 5), outputs=outputs, padding=0, strides=(1, 1), argmax=argmax, k=k)
    assert (outputs == np.array([[[[5], [6], [7], [8]],
                                  [[5], [6], [7], [8]],
                                  [[5], [6], [7], [8]]]])).all()
    assert (argmax == np.array([[[[4], [5], [6], [7]],
                                 [[12], [13], [14], [15]],
                                 [[20], [21], [22], [23]]]])).all()


if __name__ == "__main__":
    test_sequence_k1()
