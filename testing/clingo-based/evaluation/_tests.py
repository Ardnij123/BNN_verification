TESTS = {
'test': [
    ('--hamming-distance', ['2']),
    ('--argmax', [
        'bnn_encoding/argmax/output_direct_01.lp',
        'bnn_encoding/argmax/output_direct_eqconst_01.lp',
        'bnn_encoding/argmax/output_potential_01.lp',
        'bnn_encoding/argmax/output_variable_01.lp',
        ]),
    ('-t', ['1','4']),
    ],

'heat': [
    ('-t', ['4']*20),
    ('--hamming-distance', ['0', '1', '2', '3', '4']),
    ],

'correct-perceptron': [
    ('--hamming-distance', ['2']),
    ('--perceptron', [
        'bnn_encoding/perceptron/direct.lp'
        'bnn_encoding/perceptron/direct_pm1.lp'
        'bnn_encoding/perceptron/potential.lp'
        ]),
    ],

'correct-argmax': [
    ('--hamming-distance', ['2']),
    ('--perceptron', [
        'bnn_encoding/argmax/max_agg.lp'
        'bnn_encoding/argmax/output_direct_01.lp'
        'bnn_encoding/argmax/output_direct_eqconst_01.lp'
        'bnn_encoding/argmax/output_potential_01.lp'
        'bnn_encoding/argmax/output_potential_pm1.lp'
        'bnn_encoding/argmax/output_variable_01.lp'
        ]),
    ],
}
