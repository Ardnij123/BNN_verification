import numpy as np
import glob
from typing import List, Generator, Tuple
from math import ceil
import os


CLINGO_PATH = "/bin/clingo"

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


# Classes for using neural network

class Block:
    bias: array_real
    weight: array_bin

    def __init__(self, folder: str):
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

        # const nodes
        const = np.multiply(
                (bn_weight == 0),
                (len(bn_weight) + 1) * (2*(bn_bias >= 0)-1)
                )

        sum_axis = np.sum(lin_weight, axis=1, keepdims=True)

        divided = np.divide(bn_weight, bn_stddev)
        sgn = -1 * np.signbit(divided) + 1 * np.signbit(-divided)

        # add to divided where is equal to zero
        # to be able to divide by it
        divided += np.multiply(divided == 0, np.ones(divided.shape))

        self.bias = (1/2) * sgn * \
            (sum_axis + bn_mean - lin_bias - np.divide(bn_bias, divided)) \
            + const

        self.weight = np.dot(np.diag(sgn.flatten()), lin_weight)


class OutputBlk(Block):
    bias: array_real
    weight: array_bin

    def __init__(self, folder: str):
        self.bias = getarray(folder+"lin_bias.csv", trans=True)
        self.weight = getarray(folder+"lin_weight.csv", dtype=np.int8)
        self.bias -= np.sum(self.weight, axis=1, keepdims=True)
        self.bias /= 2

    def args(self):
        bs = (self.bias % 1).flatten()
        ord_nodes = np.argsort(bs, kind='stable')
        # ord_nodes is sorted from the small to big precedence
        yield from ord_nodes


class NeuralNetw:
    def __init__(self, folder: str):
        inners = len(glob.glob(folder+"blk*"))
        self.network: List[InterBlk] = []
        for i in range(1, inners+1):
            self.network.append(InterBlk(f"{folder}blk{i}/"))
        self.output = OutputBlk(f"{folder}out_blk/")

    def layers(self):
        yield (0, len(self.network[0].weight[0]))
        for layer, net in enumerate(self.network + [self.output]):
            yield (layer + 1, net.layer())

    def weights(self):
        for layer, net in enumerate(self.network + [self.output]):
            for weight in net.weights():
                yield (layer,) + weight

    def biases(self):
        for layer, net in enumerate(self.network + [self.output]):
            for bias in net.biases():
                yield (layer,) + bias

    def args(self):
        for order, node in enumerate(self.output.args()):
            yield (node, order)


if __name__ == "__main__":
    folder = "models/mnist_bnn_2_blk_16_25_20_10/"
    netw = NeuralNetw(folder)

    with open("model.lp", 'w') as model:
        model.write('#include "agg.lp".\n')

        for layer in netw.layers():
            model.write(f"layer{layer}.\n")

        for weight in netw.weights():
            model.write(f"weight{weight}.\n")

        for bias in netw.biases():
            model.write(f"bias{bias}.\n")

        for outpre in netw.args():
            model.write(f"outpre{outpre}.\n")

        # Target: solution with 5 as output
        model.write(":- not output(5).")

    # show number of solutions
    os.execl(CLINGO_PATH, "-n", "0", "-q", "model.lp")
    # show one solution with parameters
    os.execl(CLINGO_PATH, "-n", "1", "model.lp")
