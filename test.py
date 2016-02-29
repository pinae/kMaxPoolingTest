#!/usr/bin/python3
# -*- coding: utf-8 -*-
from kmaxpooling import pool
import numpy as np


def test_sequence_k1():
    data = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                     [1, 2, 3, 4, 5, 6, 7, 8],
                     [1, 2, 3, 4, 5, 6, 7, 8]]).reshape((1, 3, 8, 1))
    k = (1, 1)
    argmax = np.zeros((1, 3, 4, 1, 1, 1))
    outputs = np.zeros((1, 3, 4, 1, 1, 1))
    pool(inputs=data, kernel=(1, 5), outputs=outputs, padding=0, strides=(1, 1), argmax=argmax, k=k)
    assert (outputs == np.array([[[[[[5]]], [[[6]]], [[[7]]], [[[8]]]],
                                  [[[[5]]], [[[6]]], [[[7]]], [[[8]]]],
                                  [[[[5]]], [[[6]]], [[[7]]], [[[8]]]]]])).all()
    assert (argmax == np.array([[[[[[4]]], [[[5]]], [[[6]]], [[[7]]]],
                                 [[[[12]]], [[[13]]], [[[14]]], [[[15]]]],
                                 [[[[20]]], [[[21]]], [[[22]]], [[[23]]]]]])).all()


def test_sequence_k3():
    data = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                     [1, 2, 3, 4, 5, 6, 7, 8],
                     [1, 2, 3, 4, 5, 6, 7, 8]]).reshape((1, 3, 8, 1))
    k = (1, 3)
    argmax = np.zeros((1, 3, 4, 1, 1, 3))
    outputs = np.zeros((1, 3, 4, 1, 1, 3))
    pool(inputs=data, kernel=(1, 5), outputs=outputs, padding=0, strides=(1, 1), argmax=argmax, k=k)
    assert (outputs == np.array([[[[[[3, 4, 5]]], [[[4, 5, 6]]], [[[5, 6, 7]]], [[[6, 7, 8]]]],
                                  [[[[3, 4, 5]]], [[[4, 5, 6]]], [[[5, 6, 7]]], [[[6, 7, 8]]]],
                                  [[[[3, 4, 5]]], [[[4, 5, 6]]], [[[5, 6, 7]]], [[[6, 7, 8]]]]]])).all()
    assert (argmax == np.array([[[[[[2, 3, 4]]], [[[3, 4, 5]]], [[[4, 5, 6]]], [[[5, 6, 7]]]],
                                 [[[[10, 11, 12]]], [[[11, 12, 13]]], [[[12, 13, 14]]], [[[13, 14, 15]]]],
                                 [[[[18, 19, 20]]], [[[19, 20, 21]]], [[[20, 21, 22]]], [[[21, 22, 23]]]]]])).all()


def test_sequence_k2_2():
    data = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                     [1, 2, 3, 4, 5, 6, 7, 8],
                     [1, 2, 3, 4, 5, 6, 7, 8]]).reshape((1, 3, 8, 1))
    k = (2, 2)
    argmax = np.zeros((1, 1, 6, 1, 2, 2))
    outputs = np.zeros((1, 1, 6, 1, 2, 2))
    pool(inputs=data, kernel=(3, 3), outputs=outputs, padding=0, strides=(1, 1), argmax=argmax, k=k)
    assert (outputs == np.array([[[[[[2, 3], [2, 3]]], [[[3, 4], [3, 4]]], [[[4, 5], [4, 5]]], [[[5, 6], [5, 6]]],
                                   [[[6, 7], [6, 7]]], [[[7, 8], [7, 8]]]]]])).all()
    assert (argmax == np.array([[[[[[1, 2], [9, 10]]], [[[2, 3], [10, 11]]], [[[3, 4], [11, 12]]], [[[4, 5], [12, 13]]],
                                   [[[5, 6], [13, 14]]], [[[6, 7], [14, 15]]]]]])).all()


if __name__ == "__main__":
    test_sequence_k1()
    test_sequence_k3()
    test_sequence_k2_2()
