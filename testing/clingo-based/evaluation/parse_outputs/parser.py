#!/bin/python3

import ast
import argparse
from itertools import batched
import re


columns=[
    'Parameters', 'Models', 'Time', 'Solving', 'Model_1', 'Unsat', 'CPUTime', 'WC'
]

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
                    parsed.append({'Parameters': body.strip()})
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
                parsed.pop()
                skipping = True

    return parsed


def aggregate(entries, agg_columns, operation="print"):
    if not entries:
        return []

    entries = entries.copy()
    out_entries = []

    operated = []
    for i in range(len(entries[0])):
        if i not in agg_columns:
            operated.append(i)

    while entries:
        entry = entries[0]
        match = list(map(lambda i: entry[i], agg_columns))

        matched = []
        for pos in range(len(entries)-1, -1, -1):
            other = entries[pos]
            if match == list(map(lambda i: other[i], agg_columns)):
                matched.append(entries.pop(pos))
        
        if operation == "print":
            out_entries += matched

        elif operation == "top3avg":
            if len(matched) < 3:
                continue
            matched.sort()
            matched = matched[:3]

            row = []
            for i, value in enumerate(matched[0]):
                if i in agg_columns:
                    row.append(value)
                else:
                    row.append('%.3f' % (sum(map(lambda match: match[i], matched)) / 3))
            out_entries.append(row)

    return out_entries


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
    parser.add_argument('-A', '--aggregate', nargs='*',
                        action='extend', default=[])

    args = parser.parse_args()
    entries = parse_file(args.filename)

    if args.filter:
        args.filter = batched(args.filter, 2)

    args.aggregate = list(map(int, args.aggregate))

    for par in entries:
        for key, value in batched(par['Parameters'].split(' '), 2):
            key = key.strip('-')
            par[key] = value

    for key, value in args.filter:
        entries = filter(lambda x: x[key] == value, entries)

    if args.get:
        entries = [[entry.get(key, 1e10) for key in args.get] for entry in entries]
    else:
        args.get = columns

    if args.aggregate:
        entries = aggregate(entries, args.aggregate, 'top3avg')

    print(' '.join(args.get))
    print('\n'.join(map(lambda x: ' '.join(map(str, x)), entries)))
