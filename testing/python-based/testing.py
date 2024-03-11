import timeit
import statistics as stat
from generators import base_gen, Stepin, Scramble  # noqa
import numpy as np
import cProfile


folder = "models/mnist_bnn_2_blk_16_25_20_10/"
inpsize = 16
inpbits = 16
repeat = 5
basename = "naive"
base = __import__(basename)
basenet = base.NeuralNetw(folder)


print("######### Test data #########")
print(f"model: {folder}")
print(f"inpbits: {inpbits}")
print(f"repeats: {repeat}\n")

# print("######### Testing speed of generator #########")
# command = f"""
# i = 0
# for _ in Stepin({inpsize}, {inpbits}):
#     i += 1
# """

# r = timeit.repeat(
#         command,
#         setup="from __main__ import Stepin",
#         repeat=repeat,
#         number=1
#         )
# print(f"""Mean:  {stat.mean(r)}
# StDev: {stat.pstdev(r)}
# Min:   {min(r)}""")


# for name in ["naive", "precomp", "bylayer"]:
# for name in ["binar", "partial"]:
# for name in ["naive", "precomp", "bylayer", "binar", "partial"]:
for name in ["naive"]:
    module = __import__(name)

    print(f"######### Testing module {name} #########")
    network = module.NeuralNetw(folder)

    # print(f"Testing equality to {basename} on comp")
    # for inp in Stepin(inpsize, inpbits):
    #     if basenet.comp(inp) != network.comp(inp):
    #         print("Not equal on input:")
    #         print(inp.flatten())
    #         print(f"{basename} value: {basenet.comp(inp)}")
    #         print(f"{name} value: {network.comp(inp)}")
    #         print("")
    #         break
    # else:
    #     print(f"Module is equal to {basename}\n")

    # print(f"Testing equality to {basename} on test")
    # if np.array_equal(
    #         basenet.test(Stepin(inpsize, inpbits)),
    #         network.test(Stepin(inpsize, inpbits))
    #         ):
    #     print(f"Module is equal to {basename}\n")
    # else:
    #     print(f"Not equal to {basename}\n")

    # print(network.test(Scramble(inpsize, inpbits, scramble=True)))
    print(network.test(Stepin(inpsize, inpbits)))

    # print("Testing speed")
    # r = timeit.repeat(
    #         f"network.test(Stepin({inpsize}, {inpbits}))",
    #         setup="from __main__ import network, module, base_gen, Stepin",
    #         repeat=repeat,
    #         number=1
    #         )
    # print(f"""Mean:  {stat.mean(r)}
# StDev: {stat.pstdev(r)}
# Min:   {min(r)}\n""")

"""
    print("Profiling by cProfile")
    s = Stepin(inpsize, inpbits)
    cProfile.run("network.test(s)", sort='cumtime')
"""
