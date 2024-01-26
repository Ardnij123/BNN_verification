import numpy as np
import glob
from typing import List, Tuple
import math
from generators import Stepin


# Type annotations

array_bin = np.ndarray
array_real = np.ndarray
array_int = np.ndarray


# Working with arrays

# dot product over uint8-packed bits
def bin_dot(inp, weight):
    return bit_count(weight & inp).sum(axis=1, keepdims=True)


# returns min and max possible combinations
# with given unset bits
# Always output_minus <= output_plus
def layer_lin(
        unset_bits: array_bin,
        inp: array_bin,
        weight_plus: array_bin,
        weight_minus: array_bin,
        bias: array_real) -> Tuple[array_real, array_real]:
    base = bias + bin_dot(inp, weight_plus) - bin_dot(inp, weight_minus)
    plus = bin_dot(unset_bits, weight_plus)
    minus = bin_dot(unset_bits, weight_minus)
    return base - minus, base + plus


# Takes in inp_min and inp_max from layer_lin
# returns unset_bits and inp for next layer
#
# unset_bit == 1 iff setting unset bits from prev layers
#   could yield both 1, 0
def layer_bin(
        inp_min: array_real,
        inp_max: array_real,
        uint8_len=None) -> Tuple[array_bin, array_bin]:
    # s = 1  if 0       <= inp_min <= inp_max
    # s = 0  if inp_min <  0       <= inp_max
    # s = -1 if inp_min <= inp_max <  0
    s = 1 - np.signbit(inp_min) - np.signbit(inp_max)
    unset = (s == 0)
    sett = (s + 1) // 2
    unset_p = pack_uint8(unset, vector=True)
    sett_p = pack_uint8(sett, vector=True)
    if uint8_len is not None:
        unset_p.resize(uint8_len)
        sett_p.resize(uint8_len)
    return unset_p, sett_p


# Takes inp_min, inp_max from layer_lin
# If there is some class s.t. min is bigger than any other max
#   then ouputs that class
# If there is no such class outputs -1
def arg_max(
        inp_min: array_real,
        inp_max: array_real):
    maxmin = np.argmax(inp_min)
    sub = inp_max - inp_min[maxmin]
    if not np.all(sub <= 0):
        return -1
    for i in range(maxmin+1, len(inp_max)):
        if sub[i] == 0:
            return -1
    return maxmin


# Taken from
# https://stackoverflow.com/a/68943135/12484699
# def bit_count(arr):
#     # Make the values type-agnostic (as long as it's integers)
#     t = arr.dtype.type
#     mask = np.iinfo(t).max
#     s55 = t(0x5555555555555555 & mask)  # Add more digits for 128bit support
#     s33 = t(0x3333333333333333 & mask)
#     s0F = t(0x0F0F0F0F0F0F0F0F & mask)
#     s01 = t(0x0101010101010101 & mask)
#
#     arr = arr - ((arr >> 1) & s55)
#     arr = (arr & s33) + ((arr >> 2) & s33)
#     arr = (arr + (arr >> 4)) & s0F
#     return (arr * s01) >> (8 * (arr.itemsize - 1))

# bit_count = np.vectorize(lambda i: bin(i).count("1"))

bit_count = np.vectorize(lambda i: i.bit_count())


# Converts array (or 2d array) of +-1 or +1,0 bits
# into array (or 2d array) of packed bits
def pack_uint8(arr, vector=False):
    arr = (arr + 1) // 2  # convert into +1,0
    shape = arr.shape
    if not vector:
        arr = np.pad(
                arr,
                tuple([(0, 0)]*(len(shape)-1)
                      + [(0, 8-shape[-1] % 8)])
                )
    packed = np.packbits(arr, bitorder='little')
    if not vector:
        packed.resize(shape[:-1] + (math.ceil(shape[-1]/8),))
    return packed


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
    # packing weight bits
    weight_plus = pack_uint8(weight == 1)
    weight_minus = pack_uint8(weight == -1)
    out_len, inp_len = weight.shape

    # adding to bias
    bias -= \
        weight.sum(axis=1, keepdims=True)

    return weight_plus, weight_minus, bias/2
