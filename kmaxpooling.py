#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np


def pool(inputs, kernel, outputs, padding, strides, argmax, k=(1, 1)):
    pool_h = kernel[0]
    pool_w = kernel[1]
    stride_x = strides[1]
    stride_y = strides[0]
    n_inputs = inputs.shape[0]
    n_channels = inputs.shape[3]
    in_h = inputs.shape[1]
    in_w = inputs.shape[2]
    out_h = outputs.shape[1]
    out_w = outputs.shape[2]
    k_h = k[0]
    k_w = k[1]
    for i in range(n_inputs):
        for c in range(n_channels):
            for y_out in range(out_h):
                y = y_out * stride_y - padding
                y_min = max(y, 0)
                y_max = min(y + pool_h, in_h)
                for x_out in range(out_w):
                    x = x_out * stride_x - padding
                    x_min = max(x, 0)
                    x_max = min(x + pool_w, in_w)

                    step1 = np.zeros((y_max-y_min, k_w))
                    step1_indexes = np.zeros((y_max-y_min, k_w))
                    step1_mins = np.zeros(y_max-y_min)
                    step1_min_local_indexes = np.zeros(y_max-y_min, dtype=int)
                    for y in range(0, y_max-y_min):
                        step1_mins[y] = inputs[i, y_min+y, x_min, c]
                        step1_min_local_indexes[y] = 0
                        for x in range(0, min(k_w, x_max-x_min)):
                            step1[y, x] = inputs[i, y_min+y, x_min+x, c]
                            step1_indexes[y, x] = ((y_min+y) * in_w + (x_min+x)) * n_channels + c
                            if step1[y, x] < step1_mins[y]:
                                step1_mins[y] = step1[y, x]
                                step1_min_local_indexes[y] = x
                        for x in range(min(k_w, x_max-x_min), x_max-x_min):
                            if inputs[i, y_min+y, x_min+x, c] > step1_mins[y]:
                                for xi in range(step1_min_local_indexes[y], k_w-1):
                                    step1[y, xi] = step1[y, xi+1]
                                    step1_indexes[y, xi] = step1_indexes[y, xi+1]
                                step1[y, k_w-1] = inputs[i, y_min+y, x_min+x, c]
                                step1_indexes[y, k_w-1] = ((y_min+y) * in_w + (x_min+x)) * n_channels + c
                                step1_mins[y] = step1[y, 0]
                                step1_min_local_indexes[y] = 0
                                for xi in range(1, k_w):
                                    if step1[y, xi] < step1_mins[y]:
                                        step1_mins[y] = step1[y, xi]
                                        step1_min_local_indexes[y] = xi
                    step2_mins = np.zeros(k_w)
                    step2_min_local_indexes = np.zeros(k_w, dtype=int)
                    for x in range(0, k_w):
                        step2_mins[x] = step1[0, x]
                        step2_min_local_indexes[x] = 0
                        for y in range(0, min(k_h, y_max-y_min)):
                            outputs[i, y_out, x_out, c, y, x] = step1[y, x]
                            argmax[i, y_out, x_out, c, y, x] = step1_indexes[y, x]
                            if step1[y, x] < step2_mins[x]:
                                step2_mins[x] = step1[y, x]
                                step2_min_local_indexes[x] = y
                        for y in range(min(k_h, y_max-y_min), y_max-y_min):
                            if step1[y, x] > step2_mins[x]:
                                for yi in range(step2_min_local_indexes[x], k_h-1):
                                    outputs[i, y_out, x_out, c, yi, x] = outputs[i, y_out, x_out, c, yi+1, x]
                                    argmax[i, y_out, x_out, c, yi, x] = argmax[i, y_out, x_out, c, yi+1, x]
                                outputs[i, y_out, x_out, c, k_h-1, x] = step1[x, y]
                                argmax[i, y_out, x_out, c, k_h-1, x] = step1_indexes[x, y]
                                step2_mins[y] = outputs[i, y_out, x_out, c, 0, x]
                                step2_min_local_indexes[y] = 0
                                for yi in range(1, k_h):
                                    if outputs[i, y_out, x_out, c, yi, x] < step2_mins[x]:
                                        step2_mins[x] = outputs[i, y_out, x_out, c, yi, x]
                                        step2_min_local_indexes[x] = yi
