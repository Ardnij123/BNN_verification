#!/bin/python3

import os
from itertools import product
import pathlib
from _tests import TESTS


output_directory = 'evaluation'


def _write(file, string):
    file.write(string)
    file.write('\n')

with open(f'{output_directory}/params.txt', 'w') as file:
    for test, params in TESTS.items():
        test_dir = f'{output_directory}/{test}'

        if pathlib.Path(test_dir).exists():
            print(f'Test <{test}> already solved, skipping')
            continue

        print(f'Adding test <{test}> to queue')
        _write(file, f'! {test_dir}')
        param_space = list(product(*map((lambda x: [[x[0], a] for a in x[1]]), params)))
        for pars in param_space:
            _write(file, ' '.join(map(' '.join, pars)))

os.execlp(f'{output_directory}/eval_params.sh', f'{output_directory}/eval_params.sh')
