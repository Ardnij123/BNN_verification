import numpy as np
import glob
from typing import List, Generator, Dict
import math


# Type annotations

array_bin = np.ndarray
array_real = np.ndarray
array_int = np.ndarray


# Creation of arrays from file

def getarray(f: str, dtype=None, shape=None, trans=False) -> np.ndarray:
    arr = np.genfromtxt(f, delimiter=',', dtype=dtype)
    if len(arr.shape) == 1:
        arr = np.reshape(arr, [1, len(arr)])
    if shape is not None:
        return arr.reshape(shape)
    if trans:
        return np.transpose(arr)
    return arr


def frombuffer_vec(buffer) -> array_int:
    arr = np.frombuffer(buffer, dtype=np.uint8)
    # arr = np.reshape(arr, [len(arr), 1])
    return arr


# Working with arrays

def layer_lin(
        inp: array_bin,
        weight: array_bin, bias: array_real) -> array_real:
    return 4*bit_count(weight & inp).sum(axis=1, keepdims=True) \
          - 2*bit_count(inp).sum() + bias


def layer_bin(inp: array_real, uint8_len=None) -> array_bin:
    s = 1 - 2*np.signbit(inp)
    p = pack_uint8(s, vector=True)
    if uint8_len is not None:
        p.resize(uint8_len)
    return p


def arg_max(inp):
    return np.argmax(inp)


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
# converts weight from +-1 to +1,0 bits
# and adjusts bias to match
def binarize_wb(weight, bias):
    # packing weight bits
    weight_bin = pack_uint8(weight)
    out_len, inp_len = weight.shape

    # adding to bias
    bias -= \
        2*bit_count(weight_bin).sum(axis=1, keepdims=True)
    bias += inp_len

    return weight_bin, bias


# Classes for using neural network

class Block:
    def __init__(self, folder: str):
        pass

    def comp(self, inp: array_bin):
        pass


class InterBlk(Block):
    def __init__(self, folder: str):
        bn_bias = getarray(folder+"bn_bias.csv", trans=True)
        bn_mean = getarray(folder+"bn_mean.csv", trans=True)
        bn_stddev = getarray(folder+"bn_var.csv", trans=True)
        bn_weight = getarray(folder+"bn_weight.csv", trans=True)
        lin_bias = getarray(folder+"lin_bias.csv", trans=True)
        lin_weight = getarray(folder+"lin_weight.csv", dtype=np.int8)

        out_len, inp_len = lin_weight.shape
        self.out_uint8_len = out_len // 8 + (1 if out_len % 8 != 0 else 0)

        weight, bias = \
            precomp_wb(lin_weight, lin_bias, bn_weight,
                       bn_bias, bn_mean, bn_stddev)

        self.weight, self.bias = binarize_wb(weight, bias)

    def comp(self, inp: array_bin) -> array_bin:
        return layer_bin(
                layer_lin(inp, self.weight, self.bias),
                self.out_uint8_len
                )


class OutputBlk(Block):
    def __init__(self, folder: str):
        bias = getarray(folder+"lin_bias.csv", trans=True)
        weight = getarray(folder+"lin_weight.csv", dtype=np.int8)

        self.weight, self.bias = binarize_wb(weight, bias)

    def comp(self, inp: array_bin):
        return arg_max(
                layer_lin(inp, self.weight, self.bias)
                )


class NeuralNetw:
    def __init__(self, folder: str):
        inners = len(glob.glob(folder+"blk*"))
        self.network: List[InterBlk] = []
        for i in range(1, inners+1):
            self.network.append(InterBlk(f"{folder}blk{i}/"))
        self.output = OutputBlk(f"{folder}out_blk/")

    def comp(self, inp: array_bin) -> np.int8:
        # Do not use this to assess model
        inp = pack_uint8(inp, vector=True)
        for net in self.network:
            inp = net.comp(inp)
        return self.output.comp(inp)

    def test(self, generator: Generator[array_bin, None, None]) -> array_int:
        freq: Dict[bytes, int] = dict()
        for inp in generator:
            freq[pack_uint8(inp, vector=True).tobytes()] = 1
        for net in self.network:
            newfreq: Dict[bytes, int] = dict()
            for inpb, mult in freq.items():
                inp = frombuffer_vec(inpb)
                out = net.comp(inp)
                outb = out.tobytes()
                newfreq[outb] = mult + newfreq.get(outb, 0)
            freq = newfreq
        frequencies = np.zeros(self.output.bias.shape, dtype=np.int64)
        for inpb, mult in freq.items():
            inp = frombuffer_vec(inpb)
            outclass = self.output.comp(inp)
            frequencies[outclass] = mult + frequencies[outclass]

        return frequencies
