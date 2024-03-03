import numpy as np
import glob
from typing import List
from generators import Stepin

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


# Working with arrays

def layer_lin(
        inp: array_bin,
        weight: array_bin,
        bias: array_real) -> array_real:
    return np.dot(weight, inp) + bias


def layer_bn(
        inp: array_real,
        weight: array_real,
        bias: array_real,
        mean: array_real,
        stddev: array_real) -> array_real:
    return np.multiply(
                np.divide(weight, stddev),
                (inp - mean)
            ) + bias


def layer_bin(inp: array_real) -> array_bin:
    return 1 - 2*np.signbit(inp)


def arg_max(inp):
    return np.argmax(inp)


# Classes for using neural network

class Block:
    def __init__(self, folder: str):
        pass

    def comp(self, inp: array_bin):
        pass


class InterBlk(Block):
    def __init__(self, folder: str):
        self.bn_bias = getarray(folder+"bn_bias.csv", trans=True)
        self.bn_mean = getarray(folder+"bn_mean.csv", trans=True)
        self.bn_stddev = getarray(folder+"bn_var.csv", trans=True)
        self.bn_weight = getarray(folder+"bn_weight.csv", trans=True)
        self.lin_bias = getarray(folder+"lin_bias.csv", trans=True)
        self.lin_weight = getarray(folder+"lin_weight.csv", dtype=np.int8)

    def comp(self, inp: array_bin) -> array_bin:
        return layer_bin(
                layer_bn(
                    layer_lin(inp, self.lin_weight, self.lin_bias),
                    self.bn_weight, self.bn_bias,
                    self.bn_mean, self.bn_stddev
                    )
                )


class OutputBlk(Block):
    def __init__(self, folder: str):
        self.lin_bias = getarray(folder+"lin_bias.csv", trans=True)
        self.lin_weight = getarray(folder+"lin_weight.csv", dtype=np.int8)

    def comp(self, inp: array_bin) -> np.int8:
        return arg_max(
                layer_lin(inp, self.lin_weight, self.lin_bias)
                )


class NeuralNetw:
    def __init__(self, folder: str):
        inners = len(glob.glob(folder+"blk*"))
        self.network: List[InterBlk] = []
        for i in range(1, inners+1):
            self.network.append(InterBlk(f"{folder}blk{i}/"))
        self.output = OutputBlk(f"{folder}out_blk/")

    def comp(self, inp: array_bin) -> np.int8:
        for net in self.network:
            inp = net.comp(inp)
        return self.output.comp(inp)

    def _comp(self, inp: array_bin) -> np.int8:
        for net in self.network:
            inp = net.comp(inp)
        return self.output.comp(inp)

    def test(self, generator: Stepin) -> array_int:
        frequencies = np.zeros(self.output.lin_bias.shape, dtype=np.int64)
        for inp in generator:
            frequencies[self._comp(inp)] += 1
        return frequencies
