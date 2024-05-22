#!/usr/bin/python3

import sys

with open(sys.argv[1], 'r') as f:
    for line in f:
        if line[:4] == 'TEST':
            print(line)
        elif line[:6] == 'Model ':
            print(line[:-1])
        elif line[:6] == 'Models':
            M = line.split(' ')[-1][:-1]
        elif line[:4] == 'Time':
            times = line[15:].split(' ')
            tM = times[0]
            t1 = times[5]
        elif line[:12] == 'Aspif stats:':
            cnt = 0
            for foo in line[12:].split(' '):
                if foo:
                    if cnt == 0:
                        rules = foo
                        cnt = 1
                    else:
                        words = foo
                        break
            print(f"{M} & {tM} & {t1} & {rules} & {words}\n")
