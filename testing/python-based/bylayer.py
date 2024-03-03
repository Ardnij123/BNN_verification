import numpy as np
import glob
from typing import List, Generator, Dict

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
    arr = np.frombuffer(buffer, dtype=np.int8)
    arr = np.reshape(arr, [len(arr), 1])
    return arr


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
    return 1 - 2*np.signbit(inp).astype(np.int8)


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
        bn_bias = getarray(folder+"bn_bias.csv", trans=True)
        bn_mean = getarray(folder+"bn_mean.csv", trans=True)
        bn_stddev = getarray(folder+"bn_var.csv", trans=True)
        bn_weight = getarray(folder+"bn_weight.csv", trans=True)
        lin_bias = getarray(folder+"lin_bias.csv", trans=True)
        lin_weight = getarray(folder+"lin_weight.csv", dtype=np.int8)

        self.bias = np.zeros(bn_weight.shape, dtype=np.float_)

        # if bn_weight = 0, then the node is constant
        self.bias += np.multiply(
                bn_weight == 0,
                (len(bn_weight)+1) * (2*(bn_bias < 0)-1)
                )

        dev_weight = np.divide(bn_stddev, bn_weight)
        c = bn_mean \
            - lin_bias \
            - np.multiply(dev_weight, bn_bias)

        # if bn_stddev/bn_weight < 0,
        # then negate respective row in weight and const
        inv_neg = np.diag((2 * (dev_weight >= 0) - 1).flatten())

        c = np.dot(inv_neg, c)
        lin_weight = np.dot(inv_neg, lin_weight)

        self.bias += np.multiply(
                bn_weight != 0,
                c
                )
        self.weight = lin_weight

    def comp(self, inp: array_bin) -> array_bin:
        return layer_bin(
                layer_bn(
                    layer_lin(inp, self.weight),
                    self.bias
                    )
                )


class OutputBlk(Block):
    def __init__(self, folder: str):
        self.lin_bias = getarray(folder+"lin_bias.csv", trans=True)
        self.lin_weight = getarray(folder+"lin_weight.csv", dtype=np.int8)

    def comp(self, inp: array_bin) -> np.int8:
        return arg_max(
                layer_lin(inp, self.lin_weight) + self.lin_bias
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
        for net in self.network:
            inp = net.comp(inp)
        return self.output.comp(inp)

    def test(self, generator: Generator[array_bin, None, None]) -> array_int:
        # print(generator.__next__())  # OK
        freq: Dict[bytes, int] = dict()
        for inp in generator:
            freq[inp.tobytes()] = 1
        # print(freq)  # OK
        for net in self.network:
            newfreq: Dict[bytes, int] = dict()
            for inpb, mult in freq.items():
                inp = frombuffer_vec(inpb)
                out = net.comp(inp)
                outb = out.tobytes()
                newfreq[outb] = mult + newfreq.get(outb, 0)
            freq = newfreq
        frequencies = np.zeros(self.output.lin_bias.shape, dtype=np.int64)
        for inpb, mult in freq.items():
            inp = frombuffer_vec(inpb)
            outclass = self.output.comp(inp)
            frequencies[outclass] = mult + frequencies[outclass]

        return frequencies
