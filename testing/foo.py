import binar as b
import numpy as np
import timeit
import statistics as stat


def foo():
    return


def bar():
    return 1+1


bit_count = np.vectorize(lambda i: i.bit_count())


def bin_dot(inp, weight):
    return bit_count(weight & inp).sum(axis=1, keepdims=True)


def bin_dot_copy(inp, weight):
    rows, cols = weight.shape
    sq = np.repeat(inp, rows, axis=0)
    sq.resize(weight.shape)
    return bit_count(weight & sq).sum(axis=1, keepdims=True)


inp = np.array([129, 1, 39], dtype=np.uint8)
weight = np.array(
        [
            [38, 203, 48],
            [9, 193, 82],
            [3, 49, 183]
        ],
        dtype=np.uint8)


def print_stat(s):
    print(
        f"""Mean: {stat.mean(s)}
StDev: {stat.pstdev(s)}
Min: {min(s)}
"""
    )


t = timeit.timeit("a=1+1", setup="from __main__ import foo")
print(t)
t = timeit.timeit("a=bar()", setup="from __main__ import bar")
print(t)
t = timeit.repeat(
        "bin_dot(inp, weight)",
        setup="from __main__ import bin_dot, inp, weight",
        repeat=100,
        number=1000
        )
print_stat(t)
t = timeit.repeat(
        "bit_count(weight & inp).sum(axis=1, keepdims=True)",
        setup="from __main__ import bit_count, inp, weight",
        repeat=100,
        number=1000
        )
print_stat(t)
t = timeit.repeat(
        "bin_dot_copy(inp, weight)",
        setup="from __main__ import bin_dot_copy, inp, weight",
        repeat=1000,
        number=100
        )
print_stat(t)


crr = np.array(
        [[1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1]])

inprr = np.array([1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
inprr.resize((12, 1))

print(b.pack_uint8(crr))
print(b.pack_uint8(inprr))

inprr = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
print(b.pack_uint8(inprr))
