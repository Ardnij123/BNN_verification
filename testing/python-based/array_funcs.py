import numpy as np


# Type annotations

array_bin = np.ndarray
array_real = np.ndarray
array_int = np.ndarray


# Working with arrays

# dot product over bits
def bin_dot(inp, weight):
    return np.dot(inp, weight)


# Working with arrays

def layer_lin(
        inp: array_bin,
        weight: array_bin) -> array_int:
    return np.dot(weight, inp)


def layer_bn(
        inp: array_int,
        bias: array_real) -> array_real:
    return inp - bias


def layer_bin(inp: array_real) -> array_bin:
    return 1 - 2*np.signbit(inp)


def arg_max(inp):
    return np.argmax(inp)


# Precomputes parameters into single weight and bias
# lin_weight: +-1 matrix n x m
# lin_bias: real vector m
# bn_weight: real vector m
# bn_bias: real vector m
# bn_mean: real vector m
# bn_stddev: real vector m
# only true when comparing to 0
def precomp_wb(lin_weight, lin_bias, bn_weight,
               bn_bias, bn_mean, bn_stddev):
    bias = np.zeros(bn_weight.shape, dtype=np.float_)

    # if bn_weight = 0, then the node is constant
    bias += np.multiply(
            bn_weight == 0,
            bn_bias
            )
    lin_weight = np.dot(
            np.diag((bn_weight != 0).flatten()),
            lin_weight
            )

    dev_weight = np.divide(bn_stddev, bn_weight)
    c = - bn_mean \
        + lin_bias \
        + np.multiply(dev_weight, bn_bias)

    # if bn_stddev/bn_weight < 0,
    # then negate respective row in weight and bias
    inv_neg = np.diag((2 * (dev_weight >= 0) - 1).flatten())

    c = np.dot(inv_neg, c)
    weight = np.dot(inv_neg, lin_weight)

    bias += np.multiply(
            bn_weight != 0,
            c
            )
    # Here <inp, W> + bias >= 0
    return weight, bias


# binarizes weight matrix and bias vector
# converts weight from +-1,0 to +1,0 matrixes
# one for +1, other for -1
# and adjusts bias to match
def binarize_wb(weight, bias):
    pass
#    # packing weight bits
#    weight_plus = pack_uint8(weight == 1)
#    weight_minus = pack_uint8(weight == -1)
#    out_len, inp_len = weight.shape
#
#    # adding to bias
#    bias += \
#        weight.sum(axis=1, keepdims=True)
#
#    return weight_plus, weight_minus, bias/2
