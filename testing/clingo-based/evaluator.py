#!/bin/python

from clingo_file import NeuralNetw, Hamming, Inpbits, getarray
import os
import argparse
import re


'''
TODOs:
    output/1 constraints
'''


################################
#         Default values       #
################################

######### input params #########

# bnn to encode
BNN_dir = "models/mnist_bnn_2_blk_100_50_20_10/"

# used encoding
base = "bnn_encoding/base.lp"
perceptron = "bnn_encoding/perceptron/direct.lp"
argmax = "bnn_encoding/argmax/output_direct_01.lp"

# constraints on input region
input_base = "inputs/instance_9_100.txt"
input_hamming = "bnn_encoding/input_region/hamming.lp"
input_fixed_bits = "bnn_encoding/input_region/fixed_bits.lp"

# constraints on output
output_constraint = None

######## output params #########
intermediate_bnn = "outputs/bnn_encoded.lp"
intermediate_instance = "outputs/bnn_instance.lp"
intermediate_problem = "outputs/problem.lp"

######## Environment ###########
clingo_path = 'clingo'


################################
#        Parser of input       #
################################

parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter
        )

# BNN encoding
parser.add_argument(
        '-m', '--model', action='store', default=BNN_dir,
        help='''Directory of the BNN model
        '''
        )

# Encoding of computation
parser.add_argument(
        '-B', '--base', action='store', default=base,
        help='''Base of the encoding
        '''
        )
parser.add_argument(
        '-P', '--perceptron', action='store', default=perceptron,
        help='''Encoding of inner layer computation
        '''
        )
parser.add_argument(
        '-A', '--argmax', action='store', default=argmax,
        help='''Encoding of argmax layer computation
        '''
        )
parser.add_argument(
        '-H', '--hamming-encoding', action='store', default=input_hamming,
        help='''Encoding of hamming distance constraint
This parameter is not used when --hamming-distance is not set
        '''
        )
parser.add_argument(
        '-F', '--fixed-bits-encoding', action='store', default=input_fixed_bits,
        help='''Encoding of fixed bits distance constraint
This parameter is not used when --fixed-bits is not set
        '''
        )

# Constraints on input
parser.add_argument(
        '-i', '--input-base', action='store', default=input_base,
        help='''File with base input for the constraining of input region
Its size should correspond to the used --base
        '''
        )
parser.add_argument(
        '-d', '--hamming-distance', action='store', default=None,
        help='''Sets the maximal hamming distance from --input-base
        '''
        )
parser.add_argument(
        '-f', '--fixed-bits', action='store', default=None,
        help='''File with positions of fixed bits
Values of fixed bits are specified by --input-base
        '''
        )

# Intermediate files
parser.add_argument(
        '-x', '--intermediate-bnn', action='store', default=intermediate_bnn,
        help='''Where to store intermediate encoding of BNN
        '''
        )
parser.add_argument(
        '-y', '--intermediate-instance', action='store', default=intermediate_instance,
        help='''Where to store intermediate encoding of problem instance
        '''
        )
parser.add_argument(
        '-z', '--intermediate-problem', action='store', default=intermediate_problem,
        help='''Where to store intermediate encoding of robustness problem
        '''
        )

# Output, solutions, Clingo parameters
parser.add_argument(
        '-L', '--logic-program', action='store_true', default=False,
        help='''Only create the clingo logic program and exit
        '''
        )
parser.add_argument(
        '-p', '--print', action='extend', nargs='+', default=None,
        help='''Choose how to print solution, by default do not print anything
Multiple values may be specified using '-p value1 -p value2...'

Possible values:
    output - print output of each solution
    on - print all intermediate values of neurons
    input - print input values
        '''
        )
parser.add_argument(
        '-S', '--solutions', action='store', default=0,
        help='''How many solutions to find, 0 for all solutions
This is useful when only one solution is needed
        ''', type=int
        )
parser.add_argument(
        '-C', '--clingo', action='store', default=clingo_path,
        help='''Path to Clingo binary
        '''
        )
parser.add_argument(
        '-T', '--time-limit', action='store', default=0, type=int,
        help='''Time limit for solving
Set to 0 (default value) for no time limit
        '''
        )
parser.add_argument(
        '-t', '--parallel-threads', action='store', default=1, type=int,
        help='''Number of threads to use in search
        '''
        )

# Constraints on solution
parser.add_argument(
        '-c', '--constraint', action='extend', nargs='+', default=None,
        help='''Adds constraints on solution

By default the only constraint is on output not being the same as output of the base input:
    ':- output({b_output()}).'
By using any --constraint argument, the default constraint is not used

Special symbols:
    {b_output()} - output of base input
    {b_on(L,N)}  - equal to 1 if perceptron N in layer L is active when base input is active
        '''
        )

args = parser.parse_args()


################################
#       Encoding problem       #
################################

def get_headers(file):
    with open(file, 'r') as f:
        headers = {}
        for line in f:
            if len(line) >= 3 and line[2] == '!':
                head, value = line[3:-1].split(' ')
                headers[head] = value
            else:
                break
    return headers

def include(model, file):
    # headers = get_headers(file)
    model.write(f"#include \"{file}\".\n")


# Create encoding of BNN structure
with open(args.intermediate_bnn, 'w') as model_bnn:
    argmax_bin = get_headers(args.argmax)['binarised']
    inner_bin = get_headers(args.perceptron)['binarised']

    netw = NeuralNetw(args.model, inner_bin=='01', argmax_bin=='01')

    for layer in netw.layers():
        model_bnn.write("layer(%d, %d).\n" % layer)

    for weight in netw.weights():
        model_bnn.write(f"weight(%d, %d, %d, %d).\n" % weight)

    for bias in netw.biases():
        model_bnn.write(f"bias(%d, %d, %d).\n" % bias)

    for outpre in netw.args():
        model_bnn.write(f"outpre(%d, %d).\n" % outpre)


# Create encoding of problem instance
if args.hamming_distance is None and args.fixed_bits is None:
    raise AttributeError("Evaluator.py: Exactly one of attributes '--hamming-distance', '--fixed-bits' must be choosen.")
if args.hamming_distance is not None and args.fixed_bits is not None:
    raise AttributeError("Evaluator.py: Exactly one of attributes '--hamming-distance', '--fixed-bits' must be choosen.")

def parse_constraint(constraint, activity, output):
    expressions = re.findall(r'{[^}]*}', constraint)
    for expression in expressions:
        if expression == '{b_output()}':
            value = output
        else:
            L, N = re.findall(r'\d+', expression)
            value = activity[int(L)][int(N)-1][0]
        constraint = constraint.replace(expression, str(value), 1)
    return constraint

if not args.constraint:
    args.constraint = [':- output({b_output()}).']
constraints = []
netw = NeuralNetw(args.model)
activity, output = netw.comp_all(getarray(args.input_base, delimiter=' ', shape=[-1,1]))
for constraint in args.constraint:
    constraints.append(parse_constraint(constraint, activity, output))

with open(args.intermediate_instance, 'w') as instance:
    if args.hamming_distance:
        include(instance, args.hamming_encoding)
        instance.write(Hamming(args.input_base, args.hamming_distance).get())
    elif args.fixed_bits:
        include(instance, args.fixed_bits_encoding)
        instance.write(Inpbits(args.input_base, args.fixed_bits).get())
    else:
        raise Exception("Evaluator.py: Problem with input of problem instance.")

    for constraint in constraints:
        instance.write(constraint + '\n')


# Create encoding of problem
with open(args.intermediate_problem, 'w') as model:
    # BNN computation
    include(model, args.base)
    include(model, args.perceptron)
    include(model, args.argmax)

    # BNN structure
    include(model, args.intermediate_bnn)

    # Input of problem instance
    include(model, args.intermediate_instance)

    # Print
    model.write('#show.')
    if args.print:
        for value in args.print:
            if value == 'output':
                model.write('#show output/1.')
            elif value == 'on':
                model.write('#show on/2.')
            elif value == 'input':
                model.write('#show input(N) : on(0, N).')
            else:
                raise Exception(f"Evaluator.py: Unexpected value of --output encountered: {value}.")


################################
#              Run             #
################################

# Only create the logic program
if args.logic_program:
    exit(0)

arguments = [args.clingo]
arguments.extend(["-n", str(args.solutions)])
arguments.extend(["--time-limit", str(args.time_limit)])
arguments.extend(["--parallel-mode", str(args.parallel_threads)])
if not args.print:
    arguments.extend(['-q'])
arguments.extend([args.intermediate_problem])

os.execlp(*arguments)
