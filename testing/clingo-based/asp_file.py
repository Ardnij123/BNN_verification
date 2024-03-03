import numpy as np
import glob
from typing import List, Generator, Tuple
import clingo
from math import ceil

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


# Classes for using neural network

class Block:
    bias: array_real
    weight: array_bin

    def __init__(self, folder: str):
        pass

    def comp(self, inp: array_bin):
        pass

    def layer(self) -> int:
        return len(self.bias)

    def weights(self) -> Generator[Tuple[int, int, int], None, None]:
        for rown, row in enumerate(self.weight):
            for coln, weight in enumerate(row):
                yield (coln, rown, weight)

    def biases(self) -> Generator[Tuple[int, int], None, None]:
        for node, bias in enumerate(self.bias):
            yield (node, ceil(bias[0]))


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

        self.bias += np.sum(self.weight, axis=1, keepdims=True)
        self.bias /= 2

    def comp(self, inp: array_bin) -> array_bin:
        # NOT CORRECT FOR PY IMPLEMENTATION
        return layer_bin(
                layer_bn(
                    layer_lin(inp, self.weight),
                    self.bias
                    )
                )


class OutputBlk(Block):
    bias: array_real
    weight: array_bin

    def __init__(self, folder: str):
        self.bias = getarray(folder+"lin_bias.csv", trans=True)
        self.weight = getarray(folder+"lin_weight.csv", dtype=np.int8)
        self.bias += np.sum(self.weight, axis=1, keepdims=True)
        self.bias /= 2

    def comp(self, inp: array_bin) -> np.int8:
        # NOT CORRECT FOR PY IMPLEMENTATION
        return arg_max(
                layer_lin(inp, self.weight) + self.bias
                )

    def args(self) -> List[int]:
        bs = (self.bias % 1).flatten()
        args = np.argsort(bs, kind='stable')
        revargs = [0] * len(args)
        for num, arg in enumerate(args):
            revargs[arg] = num
        return revargs


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

    def test(self, generator: Generator[array_bin, None, None]) -> array_int:
        frequencies = np.zeros(self.output.bias.shape, dtype=np.int64)
        for inp in generator:
            frequencies[self.comp(inp)] += 1
        return frequencies

    def layers(self):
        for layer, net in enumerate(self.network + [self.output]):
            yield clingo.Tuple_(
                    [clingo.Number(i) for i in (layer, net.layer())]
                )

    def weights(self):
        for layer, net in enumerate(self.network + [self.output]):
            for weight in net.weights():
                yield clingo.Tuple_(
                        [clingo.Number(i) for i in (layer,) + weight]
                        )

    def biases(self):
        for layer, net in enumerate(self.network + [self.output]):
            for bias in net.biases():
                yield clingo.Tuple_(
                        [clingo.Number(i) for i in (layer,) + bias]
                        )

    def args(self):
        for pos, arg in enumerate(self.output.args()):
            yield clingo.Tuple_(
                    [clingo.Number(i) for i in (arg, pos)]
                    )


if __name__ == "__main__":
    folder = "models/mnist_bnn_2_blk_100_100_50_10/"
    ctl = clingo.Control()
    ctl.add("base", [], """\
    #include "agg.lp".

    layer(L, N) :- (L, N) = @layers().
    weight(F, N, T, W) :- (F, N, T, W) = @weights().
    bias(L, N, B) :- (L, N, B) = @biases().
    outpre(N, O) :- (N, O) = @args().
    """)
    ctl.ground([("base", [])], context=NeuralNetw(folder))
    ctl.configuration.solve.models = "0"

    models = 0
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            models += 1
            if models % 10000 == 0:
                print(f"found {models}")
    print(f"Models: {models}")
