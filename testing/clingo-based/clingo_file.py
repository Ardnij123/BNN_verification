import numpy as np
import glob
from typing import List, Generator, Tuple
from math import ceil, floor
import os
from collections.abc import Iterable


CLINGO_PATH = "clingo"

# Type annotations

array_bin = np.ndarray
array_real = np.ndarray
array_int = np.ndarray


# Creation of arrays from file

def getarray(f: str, dtype=None, shape=None, trans=False,
             delimiter=',') -> np.ndarray:
    arr = np.genfromtxt(f, delimiter=delimiter, dtype=dtype)
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
        for rown, row in enumerate(self.weight, start=1):
            for coln, weight in enumerate(row, start=1):
                yield (coln, rown, weight)

    def biases(self) -> Generator[Tuple[int, int], None, None]:
        for node, bias in enumerate(self.bias, start=1):
            yield (node, floor(bias[0]))


class InterBlk(Block):

    def __init__(self, folder: str, binarised_01=True):
        self.folder = folder
        bn_bias = getarray(folder+"bn_bias.csv", trans=True)
        bn_mean = getarray(folder+"bn_mean.csv", trans=True)
        bn_stddev = np.sqrt(getarray(folder+"bn_var.csv", trans=True))
        bn_weight = getarray(folder+"bn_weight.csv", trans=True)
        lin_bias = getarray(folder+"lin_bias.csv", trans=True)
        lin_weight = getarray(folder+"lin_weight.csv", dtype=np.int8)

        divided = np.divide(bn_weight, bn_stddev)
        sgn = -1 * (divided < 0) + 1 * (divided > 0)

        # Lemma 2.1.2 - encoding of bnn using weight matrix and bias vector
        self.bias = np.where(divided == 0,
                bn_bias,
                sgn * (lin_bias - bn_mean + np.divide(bn_bias, divided)))

        self.weight = np.dot(np.diag(sgn.flatten()), lin_weight)

        # Section 4.2.1 - mapping input to values {0, 1}
        if binarised_01:
            sum_axis = np.sum(lin_weight, axis=1, keepdims=True)
            self.bias = (self.bias - sum_axis) / 2

    def comp(self, vector: array_bin):
        return (np.dot(self.weight, vector) + self.bias) >= 0


class OutputBlk(Block):

    def __init__(self, folder: str, binarised_01=True):
        self.folder = folder
        self.bias = getarray(folder+"lin_bias.csv", trans=True)
        self.weight = getarray(folder+"lin_weight.csv", dtype=np.int8)

        # Section 4.2.2 - mapping input to values {0, 1}
        if binarised_01:
            sum_axis = np.sum(self.weight, axis=1, keepdims=True)
            self.bias = (self.bias - sum_axis) / 2

        bs = self.bias % 1
        self.precedence = np.argsort(-bs.flatten(), kind='stable')

    def args(self):
        return self.precedence

    def comp(self, vector: array_bin):
        output = np.argmax(np.dot(self.weight, vector) + self.bias)
        return output + 1


class NeuralNetw:
    def __init__(self, folder: str, inner_01=True, argmax_01=True):
        inners = len(glob.glob(folder+"blk*"))
        self.network: List[InterBlk] = []
        for i in range(1, inners+1):
            self.network.append(InterBlk(f"{folder}blk{i}/", inner_01))
        self.output = OutputBlk(f"{folder}out_blk/", argmax_01)

    def layers(self):
        yield (0, len(self.network[0].weight[0]))
        for layer, net in enumerate(self.network + [self.output], start=1):
            yield (layer, net.layer())

    def weights(self):
        for layer, net in enumerate(self.network + [self.output], start=1):
            for weight in net.weights():
                yield (layer,) + weight

    def biases(self):
        for layer, net in enumerate(self.network + [self.output], start=1):
            for bias in net.biases():
                yield (layer,) + bias

    def args(self):
        for order, node in enumerate(self.output.args(), start=1):
            yield (node+1, order)

    def comp(self, vector: array_bin):
        for block in self.network:
            vector = block.comp(vector)
        return self.output.comp(vector)

    def comp_all(self, vector: array_bin):
        activity = [vector]
        for block in self.network:
            vector = block.comp(vector)
            activity.append(vector.astype(np.uint8))

        output = np.zeros(self.output.bias.shape, dtype=np.uint8)
        output
        output_val = self.output.comp(vector)
        output[output_val-1] = 1

        activity.append(output)
        return activity, output_val


class Constraint:
    inpbits: Iterable[int]

    def inpbits_gen(self):
        for idx, inpbit in enumerate(self.inpbits, start=1):
            if inpbit >= 0.5:
                yield f"input({idx})."
            else:
                continue
                # yield f"input({idx}, -1)."

    def get(self) -> str:
        pass


class Hamming(Constraint):
    def __init__(self, input_base: str, maxdist: int):
        self.inpbits = getarray(input_base, dtype=int, shape=[-1], delimiter=' ')
        self.hamdist = maxdist

    def hamdist_gen(self):
        return f"hammdist({self.hamdist})."

    def get(self):
        return "\n".join(self.inpbits_gen()) + '\n' + self.hamdist_gen()


class Inpbits(Constraint):
    def __init__(self, input_base: str, input_fixed: str):
        self.inpbits = getarray(input_base, dtype=int, shape=[-1], delimiter=' ')
        self.fixed_idx = getarray(input_fixed, dtype=int, shape=[-1])

    def fixed_gen(self):
        for fix in self.fixed_idx:
            yield f"inpfix({fix})."

    def get(self):
        return '\n'.join(self.inpbits_gen()) + '\n' \
               + '\n'.join(self.fixed_gen())


if __name__ == "__main__":
    folder = "models/mnist_bnn_2_blk_100_50_20_10/"
    netw = NeuralNetw(folder)

    vector = getarray("inputs/instance_0_100.txt",
                      delimiter=' ', shape=[-1, 1])
    print(f"Desired output: {netw.comp(vector)}")

    with open("model.lp", 'w') as model:
        model.write('#include "agg_2.lp".\n')

        for layer in netw.layers():
            model.write(f"layer{layer}.\n")

        for weight in netw.weights():
            model.write(f"weight{weight}.\n")

        for bias in netw.biases():
            model.write(f"bias{bias}.\n")

        for outpre in netw.args():
            model.write(f"outpre{outpre}.\n")

        model.write(Hamming(vector, 3).get())
        # model.write(Hamming([1,1,0,1,0,0,1,0,1,1,1,0,1,0,1,1], 12).get())
        # model.write(Inpbits([1,1,0,1,0,0,1,0,1,1,1,0,1,0,1,1], range(8)).get())

        # Target: solution with 5 as output
        # model.write(":- not output(5).")

    # show number of solutions
    os.execlp(CLINGO_PATH, "-n", "0", "-t", "4", "model.lp")
    #os.execlp(CLINGO_PATH, "-n", "0", "-q", "-t", "4", "model.lp")
    # show one solution with parameters
    os.execlp(CLINGO_PATH, "-n", "1", "model.lp")
