#!/bin/python3

import os
from itertools import product
import pathlib
from _tests import TESTS


output_directory = 'evaluation'


def _write(file, string, endline=False):
    file.write(string)
    if endline:
        file.write('\n')
    else:
        file.write('\\\n')

with open(f'{output_directory}/script.sh', 'w') as file:
    _write(file, '#!/bin/bash', True)

    for test, params in TESTS.items():
        test_dir = f'{output_directory}/{test}'

        if pathlib.Path(test_dir).exists():
            print(f'Test <{test}> already solved, skipping')
            continue

        print(f'Adding test <{test}> to queue')
        _write(file, f'echo "Starting <{test}>..." &&')
        _write(file, f'mkdir -p {test_dir} &&')
        param_space = list(product(*map((lambda x: [[x[0], a] for a in x[1]]), params)))
        for i, pars in enumerate(param_space, start=1):
            _write(file, f'echo "Parameters {i}/{len(param_space)}..." &&')
            _write(file, f'echo "Parameters : {pars}" >> {test_dir}/output.txt &&')
            _write(file, f'{"{"} ./evaluator.py {' '.join(map(' '.join, pars))} >> {test_dir}/output.txt || /bin/true; {"}"} &&')
            _write(file, f'echo "Word count: $( gringo outputs/problem.lp | wc)" >> {test_dir}/output.txt &&')
        _write(file, f'echo "Test <{test}> finished successfully" ||')
        # Unsuccessful run
        _write(file, f'{"{"} mv {test_dir} "{test_dir}_fail_$(date +%s)" && echo "TEST <{test}> FAILED"; {"}"}', True)

os.execlp(f'{output_directory}/script.sh', f'{output_directory}/script.sh')
