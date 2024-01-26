import numpy as np
from typing import Generator

# Type annotations

array_bin = np.ndarray
array_real = np.ndarray
array_int = np.ndarray


# Generator of inputs +- 1 of size {size}
# with permutations on first {perms} bits
def base_gen(size: int, perms: int) -> Generator[array_int, None, None]:
    inp = -np.ones([size, 1], dtype=np.int8)
    while True:
        for idx in range(size):
            if inp[idx][0] > 1:
                if idx+1 == perms:
                    return
                inp[idx][0] = -1
                inp[idx+1][0] += 2
        yield inp
        inp[0][0] += 2


# Generator of inputs +- 1 of size {size}
# with permutations on first {perms} bits
# Allows for partial generation of inputs
class Stepin:
    def __init__(self, size, perms):
        assert perms <= size
        self.unset_bits = \
            np.array([[1]]*perms + [[0]]*(size - perms), dtype=np.int8)
        self.inp = -np.ones([size, 1], dtype=np.int8)
        self.set_pos = -1
        self.perms = perms
        self.first = True

    # Iterator only on inputs
    def __iter__(self):
        return self

    def __next__(self):
        pre = self.first
        while self.set_pos != self.perms - 1:
            self.step_in()
        self.first = pre
        return self.step()[1]

    def step_in(self):
        if self.set_pos == self.perms - 1:
            raise StopIteration
        self.set_pos += 1
        self.unset_bits[self.set_pos][0] = 0
        self.first = True

    def step_out(self):
        if self.set_pos == -1:
            raise StopIteration
        self.unset_bits[self.set_pos][0] = 1
        self.inp[self.set_pos][0] = -1
        self.set_pos -= 1

    def step(self):
        if self.first:
            self.first = False
            return self.get()
        while self.inp[self.set_pos] == 1:
            self.step_out()
        if self.set_pos == -1:
            raise StopIteration
        self.inp[self.set_pos] = 1
        return self.get()

    def get(self):
        return self.unset_bits, self.inp
