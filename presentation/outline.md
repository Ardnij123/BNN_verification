# Assignment

- Theory
    - BNN robustness
    - ASP problem
- ASP encoding
    - usage of ASP solver
    - prototype implementation
- Evaluation
    - evaluation on cited data

# BNN

- near state-of-the-art results
- low energy, power needed

- DNN
    - perceptron
    - single-layer perceptron
    - multi-layer perceptron
- BNN
    - heavyside step function $H$
    - binarisation of perceptron, batch normalization
        - equivalence of BN to non-BN
    - binarised SLP, argmax layer, MLP
- Robustness
    - definition of robustness
        - input region, weight, evaluation function
        - quantitative, qualitative robustness
        - $Q=0$ for robust regions (bad formulation IG)
        - strict robustness
    - input regions
        - hamming distance
        - fixed bits
        - explicit formula for size of region
# ASP
    
- form of declarative programming

- Extended logic program
    - classical negation x negation as failure
    - syntax and semantics of logic program
    - minimality of answer set (similiar to Closed world assumption)

- Clingo
    - framework for ASP solving
    - gringo + clasp
    - enumeration of answer sets (possibility for better results?) 
    - extensions

# ASP encoding of BNN robustness

- Analysis
    - encoding BNN using integers
- Encoding of BNN
    - multiples encodings of perceptron layer, argmax layer, input regions
- Encoding of robustness
    - trivial by $\text{models}\over\text{input region size}$

# Evaluation

- Methodology
    - large portion of randomness in meassured performance
    - average of best 3 of 4
    - BNN models architectures
        - usually M1, M2, M7 except fixed bits
- Evaluation
    - of perceptron encoding
    - of output layer encoding
    - of hamming distance encoding
    - of fixed bits encoding
- Evaluation of best model

# Discussion

- more robust = lower robustness
    - according to my definition
    - would be better to say it corresponds to "brokenness" of input region

- arity as a type of function
    - was used in one of sources of this thesis [Zarba, 22]

- usage of classical negation in BNN encoding
    - IG it would not be benefiting
    - classical negation is rewritten as a constraint
    - but will implement an encoding of that
