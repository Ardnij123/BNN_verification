#!/bin/python3

import ast
import argparse
from itertools import batched
import re


def parse_file(file):
    parsed = []
    skipping = True
    with open(file, 'r') as f:
        for line in f:
            try:
                parts = line.split(':')
                head = parts[0].strip()
                body = (':'.join(parts[1:]))[:-1]
                if head == 'Parameters':
                    parsed.append({'Parameters': ast.literal_eval(body)})
                    skipping=False
                elif skipping:
                    continue
                elif head == 'Models':
                    parsed[-1]['Models'] = int(body)
                elif head == 'Time':
                    time, solving, model_1, unsat = re.match(r'\s+(\S+)\s+\S+\s+(\S+)\s+\S+\s+\S+\s+(\S+)\s+\S+\s+(\S+)\S', body).group(1, 2, 3, 4)
                    # _, time, solving, model_1, unsat = re.match(r'\w*(\W+)  \(Solving: (\W+) 1st Model: (\W+) Unsat: (\W+)\)', body)
                    parsed[-1]['Time'] = float((time.strip())[:-1])
                    parsed[-1]['Solving'] = float((solving.strip())[:-1])
                    parsed[-1]['Model_1'] = float((model_1.strip())[:-1])
                    parsed[-1]['Unsat'] = float((unsat.strip())[:-1])
                elif head == 'CPU Time':
                    parsed[-1]['CPUTime'] = float((body.strip())[:-1])
                elif head == 'Word count':
                    parsed[-1]['WC'] = body
            except Exception as e:
                print(e)
                print(head, body)
                parsed.pop()
                skipping = True

    return parsed


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter
            )

    parser.add_argument('filename')

    parser.add_argument('-F', '--filter', nargs=2,
                        action='extend', default=[],
                        help='Filters entries to only contain those with specified values.')
    parser.add_argument('-G', '--get', nargs='*',
                        action='extend', default=[])

    args = parser.parse_args()
    entries = parse_file(args.filename)

    if not args.filter:
        args.filter = batched(args.filter, 2)

    for key, value in args.filter:
        entries = filter(lambda x: x[key] == value, entries)

    if args.get:
        entries = [[entry[key] for key in args.get] for entry in entries]

    print('\n'.join(map(lambda x: ' '.join(map(str, x)), entries)))
