import sys
from math import comb

def comp_hamm(inpsize, dist):
    if dist == 0:
        return 1
    return comb(inpsize, dist) + comp_hamm(inpsize, dist-1)

def comp_fixed(inpsize, free):
    return 2**free

for line in sys.stdin.readlines():
    instance, hammdist, models = line.split(' ')
    hammdist, models = int(hammdist), int(float(models))
    _, _, size = instance[:-4].split('_')
    size = int(size)
    print('%.3f' % (100 * models / comp_fixed(size, hammdist)), models)
