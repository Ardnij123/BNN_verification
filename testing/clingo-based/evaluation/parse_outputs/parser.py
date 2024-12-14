#!/bin/python3

import ast


def parse_file(file):
    parsed = []
    with open(file, 'r') as f:

        for line in f:
            parts = line.split(':')
            head = parts[0].strip()
            body = (':'.join(parts[1:]))[:-1]
            if head == 'Parameters':
                parsed.append({'Parameters': ast.literal_eval(body)})
            elif head == 'Models':
                parsed[-1]['Models'] = body
            elif head == 'Time':
                parsed[-1]['Time'] = body
            elif head == 'CPU Time':
                parsed[-1]['CPUTime'] = body
            elif head == 'Word count':
                parsed[-1]['WC'] = body

    return parsed
