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
    ('-t', ['4']),
    ('--perceptron', [
        'bnn_encoding/perceptron/direct.lp',
        'bnn_encoding/perceptron/direct_pm1.lp',
        'bnn_encoding/perceptron/potential.lp',
        ]),
    ],

'correct-argmax': [
    ('--hamming-distance', ['2']),
    ('-t', ['4']),
    ('--argmax', [
        'bnn_encoding/argmax/max_agg.lp',
        'bnn_encoding/argmax/output_direct_01.lp',
        'bnn_encoding/argmax/output_direct_eqconst_01.lp',
        'bnn_encoding/argmax/output_potential_01.lp',
        'bnn_encoding/argmax/output_potential_pm1.lp',
        'bnn_encoding/argmax/output_variable_01.lp',
        ]),
    ],

'eval-perceptron': [
    ('-t', ['8']*4),
    ('--hamming-distance', ['3', '2', '1', '0']),
    ('-T', ['300']),
    ('--input-base', [
        'inputs/instance_0_100.txt',
        'inputs/instance_7_100.txt',
        ]),
    ('--model', [
        'models/mnist_bnn_1_blk_100_100_10/',
        'models/mnist_bnn_1_blk_100_50_10/',
        'models/mnist_bnn_2_blk_100_50_20_10/',
        ]),
    ('--perceptron', [
        'bnn_encoding/perceptron/direct.lp',
        'bnn_encoding/perceptron/direct_pm1.lp',
        #'bnn_encoding/perceptron/potential.lp',
        ]),
    ],

'eval-perceptron-potential': [
    ('--hamming-distance', ['0', '1', '2', '3']),
    ('-t', ['8']*4),
    ('-T', ['300']),
    ('--input-base', [
        'inputs/instance_0_100.txt',
        'inputs/instance_7_100.txt',
        ]),
    ('--model', [
        'models/mnist_bnn_1_blk_100_100_10/',
        'models/mnist_bnn_1_blk_100_50_10/',
        'models/mnist_bnn_2_blk_100_50_20_10/',
        ]),
    ('--perceptron', [
        #'bnn_encoding/perceptron/direct.lp',
        #'bnn_encoding/perceptron/direct_pm1.lp',
        'bnn_encoding/perceptron/potential.lp',
        ]),
    ],

'multiple-short': [
    ('-t', ['4']*1000),
    ('-T', ['5']),
    ('--hamming-distance', ['2']),
    ],

'multiple-long': [
    ('--model', ['models/mnist_bnn_3_blk_25_25_25_20_10/']),
    ('--input-base', ['inputs/instance_0_25.txt']),
    ('-t', ['8']*150),
    ('-T', ['600']),
    ('--hamming-distance', ['8']),
    ],
}
