from clingo_file import NeuralNetw, Hamming, Inpbits, getarray
import os
import argparse

# Default values

# input params
BNN_dir = "models/mnist_bnn_2_blk_100_50_20_10/"
solver = "agg_2.lp"
input_vector_file = "inputs/instance_9_100.txt"
constraint = "hamming"
hamming_dist = 2
inpbits_fixed = ""

# output params
intermediate = "model.lp"
quiet = True
solutions = 0

parser = argparse.ArgumentParser()
parser.add_argument(
        "-m", "--model", action="store", default=BNN_dir,
        help='directory of the BNN model'
        )
parser.add_argument(
        "-s", "--solver", action="store", default=solver,
        help='ASP solver to use with the model'
        )
parser.add_argument(
        "-i", "--input", action="store", default=input_vector_file,
        help='input vector used for constraining'
        )
parser.add_argument(
        "-c", "--constraint", action="store", default=constraint,
        help='type of constraint', choices=['hamming', 'inpbits']
        )
parser.add_argument(
        "-d", "--hamm-dist", action="store", default=hamming_dist,
        help='hamming distance from input vector'
        )
parser.add_argument(
        "-f", "--inpbits-fixed", action="store", default=inpbits_fixed,
        help='file with fixed input bits'
        )

parser.add_argument(
        "-x", "--intermediate", action="store", default=intermediate,
        help='where to store intermediate ASP file'
        )
parser.add_argument(
        "-p", "--print-solutions", action="store_true",
        help='print solutions'
        )
parser.add_argument(
        "-S", "--solutions", action="store", default=solutions,
        help="how many solutions to find, 0 for all solutions", type=int
        )
parser.add_argument(
        "-C", "--clingo", action="store", default="clingo",
        help="path to Clingo binary"
        )

args = parser.parse_args()

netw = NeuralNetw(args.model)

with open(args.intermediate, 'w') as model:
    model.write(f"#include \"{solver}\".\n")

    for layer in netw.layers():
        model.write(f"layer{layer}.\n")

    for weight in netw.weights():
        model.write(f"weight{weight}.\n")

    for bias in netw.biases():
        model.write(f"bias{bias}.\n")

    for outpre in netw.args():
        model.write(f"outpre{outpre}.\n")

    input_vector = getarray(args.input, dtype=int, shape=[-1], delimiter=' ')
    if args.constraint == 'hamming':
        model.write(Hamming(input_vector, args.hamm_dist).get())
    else:
        assert args.constraint == 'inpbits'
        fixed_positions = getarray(args.inpbits_fixed)
        model.write(Inpbits(input_vector, fixed_positions).get())

arguments = [args.clingo]
arguments.extend(["-n", str(args.solutions)])
if not args.print_solutions:
    arguments.extend(["-q"])
arguments.extend([args.intermediate])

os.execlp(*arguments)
