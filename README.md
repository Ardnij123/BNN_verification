This repository contains implementation of verification
of binarised neural network using ASP tool Clingo.
The implementation can be found in /testing/evaluator.py

For the explanation of this implementation, see /thesis/prace.pdf


## Minimal requirements:

Installed packages

- Clingo 5
- Python 3
    - Package NumPy
- Git (for cloning the repository)

## Quickstart:

1) Switch to directory with implementation

```bash
cd testing/
```

2) Run the evaluator

```bash
./evaluator --hamming-distance 2
```
