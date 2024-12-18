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

'eval-argmax': [
    ('--hamming-distance', ['3', '2', '1', '0']),
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
    ('--argmax', [
        #'bnn_encoding/argmax/max_agg.lp',
        'bnn_encoding/argmax/output_direct_01.lp',
        'bnn_encoding/argmax/output_direct_eqconst_01.lp',
        'bnn_encoding/argmax/output_potential_01.lp',
        'bnn_encoding/argmax/output_potential_pm1.lp',
        'bnn_encoding/argmax/output_variable_01.lp',
        ]),
    ],

'eval-argmax-max_agg': [
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
    ('--argmax', [
        'bnn_encoding/argmax/max_agg.lp',
        ]),
    ],

'eval-perceptron-inpbits': [
    ('-t', ['8']*4),
    ('-T', ['300']),
    ('--fixed-bits', [
        'fixed_bits/fixed_22.txt',
        'fixed_bits/fixed_16.txt',
        'fixed_bits/fixed_8.txt',
        'fixed_bits/fixed_all.txt',
        ]),
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

'eval-perceptron-inpbits-potential': [
    ('--fixed-bits', [
        'fixed_bits/fixed_all.txt',
        'fixed_bits/fixed_8.txt',
        'fixed_bits/fixed_16.txt',
        'fixed_bits/fixed_22.txt',
        ]),
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
        'bnn_encoding/perceptron/potential.lp',
        ]),
    ],

'eval-perceptron-inpbits-pm1': [
        # TODO: This throws segfault
    ('-t', ['8']*4),
    ('-T', ['300']),
    ('--fixed-bits', [
        'fixed_bits/fixed_16.txt',
        'fixed_bits/fixed_8.txt',
        'fixed_bits/fixed_all.txt',
        ]),
    ('--input-base', [
        'inputs/instance_0_100.txt',
        'inputs/instance_7_100.txt',
        ]),
    ('--model', [
        'models/mnist_bnn_2_blk_100_50_20_10/',
        ]),
    ('--perceptron', [
        'bnn_encoding/perceptron/direct_pm1.lp',
        ]),
    ],

'eval-argmax-inpbits': [
    ('-t', ['8']*4),
    ('-T', ['300']),
    ('--fixed-bits', [
        'fixed_bits/fixed_16.txt',
        'fixed_bits/fixed_8.txt',
        'fixed_bits/fixed_all.txt',
        ]),
    ('--input-base', [
        'inputs/instance_0_100.txt',
        'inputs/instance_7_100.txt',
        ]),
    ('--model', [
        'models/mnist_bnn_1_blk_100_100_10/',
        'models/mnist_bnn_1_blk_100_50_10/',
        'models/mnist_bnn_2_blk_100_50_20_10/',
        ]),
    ('--argmax', [
        'bnn_encoding/argmax/max_agg.lp',
        'bnn_encoding/argmax/output_direct_01.lp',
        'bnn_encoding/argmax/output_direct_eqconst_01.lp',
        'bnn_encoding/argmax/output_potential_01.lp',
        'bnn_encoding/argmax/output_potential_pm1.lp',
        'bnn_encoding/argmax/output_variable_01.lp',
        ]),
    ],

'correct-inpbits': [
    ('-t', ['8']),
    ('--model', ['models/mnist_bnn_3_blk_25_25_25_20_10/']),
    ('--input-base', ['inputs/instance_0_25.txt']),
    ('--fixed-bits', [
        'fixed_bits/fixed_all.txt',
        'fixed_bits/fixed_8.txt',
        'fixed_bits/fixed_16.txt',
        'fixed_bits/fixed_24.txt',
        ]),
    ],

'eval-input/m7': [
    ('-t', ['8']*4),
    ('-T', ['300']),
    ('--fixed-bits', [
        'fixed_bits/fixed_20.txt',
        'fixed_bits/fixed_16.txt',
        'fixed_bits/fixed_8.txt',
        'fixed_bits/fixed_all.txt',
        ]),
    ('--input-base', [
        'inputs/instance_0_100.txt',
        'inputs/instance_7_100.txt',
        ]),
    ('--model', [
        'models/mnist_bnn_2_blk_100_50_20_10/',
        ]),
    ('--fixed-bits-encoding', [
        'bnn_encoding/input_region/fixed_bits.lp',
        'bnn_encoding/input_region/fixed_direct.lp',
        'bnn_encoding/input_region/fixed_input.lp',
        ]),
    ],

'eval-input/m4': [
    ('-t', ['8']*4),
    ('-T', ['300']),
    ('--fixed-bits', [
        'fixed_bits/fixed_20_64.txt',
        'fixed_bits/fixed_16_64.txt',
        'fixed_bits/fixed_8_64.txt',
        'fixed_bits/fixed_all_64.txt',
        ]),
    ('--input-base', [
        'inputs/instance_0_64.txt',
        'inputs/instance_7_64.txt',
        ]),
    ('--model', [
        'models/mnist_bnn_1_blk_64_10_10/',
        ]),
    ('--fixed-bits-encoding', [
        'bnn_encoding/input_region/fixed_bits.lp',
        'bnn_encoding/input_region/fixed_direct.lp',
        'bnn_encoding/input_region/fixed_input.lp',
        ]),
    ],

'eval-input/m5': [
    ('-t', ['8']*4),
    ('-T', ['300']),
    ('--fixed-bits', [
        'fixed_bits/fixed_20_784.txt',
        'fixed_bits/fixed_16_784.txt',
        'fixed_bits/fixed_8_784.txt',
        'fixed_bits/fixed_all_784.txt',
        ]),
    ('--input-base', [
        'inputs/instance_0_784.txt',
        'inputs/instance_7_784.txt',
        ]),
    ('--model', [
        'models/mnist_bnn_1_blk_784_100_10/',
        ]),
    ('--fixed-bits-encoding', [
        'bnn_encoding/input_region/fixed_bits.lp',
        'bnn_encoding/input_region/fixed_direct.lp',
        'bnn_encoding/input_region/fixed_input.lp',
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

'hamming-big/25': [
    ('--hamming-distance', ['4', '3', '2', '1', '0']),
    ('-t', ['8']*4),
    ('-T', ['600']),
    ('--input-base', [
        'inputs/instance_0_25.txt',
        ]),
    ('--model', [
        'models/mnist_bnn_3_blk_25_25_25_20_10/',
        ]),
    ('--argmax', [
        'bnn_encoding/argmax/output_variable_01.lp',
        ]),
    ],

'hamming-big/36': [
    ('--hamming-distance', ['4', '3', '2', '1', '0']),
    ('-t', ['8']*4),
    ('-T', ['600']),
    ('--input-base', [
        'inputs/instance_0_36.txt',
        ]),
    ('--model', [
        'models/mnist_bnn_2_blk_36_15_10_10/',
        ]),
    ('--argmax', [
        'bnn_encoding/argmax/output_variable_01.lp',
        ]),
    ],

'hamming-big/16': [
    ('--hamming-distance', ['4', '3', '2', '1', '0']),
    ('-t', ['8']*4),
    ('-T', ['600']),
    ('--input-base', [
        'inputs/instance_0_16.txt',
        ]),
    ('--model', [
        'models/mnist_bnn_2_blk_16_25_20_10/',
        'models/mnist_bnn_3_blk_16_64_32_20_10/',
        ]),
    ('--argmax', [
        'bnn_encoding/argmax/output_variable_01.lp',
        ]),
    ],

'hamming-big/64': [
    ('--hamming-distance', ['4', '3', '2', '1', '0']),
    ('-t', ['8']*4),
    ('-T', ['600']),
    ('--input-base', [
        'inputs/instance_0_64.txt',
        ]),
    ('--model', [
        'models/mnist_bnn_1_blk_64_10_10/',
        ]),
    ('--argmax', [
        'bnn_encoding/argmax/output_variable_01.lp',
        ]),
    ],

'hamming-big/400': [
    ('--hamming-distance', ['4', '3', '2', '1', '0']),
    ('-t', ['8']*4),
    ('-T', ['600']),
    ('--input-base', [
        'inputs/instance_0_400.txt',
        ]),
    ('--model', [
        'models/mnist_bnn_1_blk_400_100_10/',
        ]),
    ('--argmax', [
        'bnn_encoding/argmax/output_variable_01.lp',
        ]),
    ],

'hamming-big/784': [
    ('--hamming-distance', ['4', '3', '2', '1', '0']),
    ('-t', ['8']*4),
    ('-T', ['600']),
    ('--input-base', [
        'inputs/instance_0_784.txt',
        ]),
    ('--model', [
        'models/mnist_bnn_1_blk_784_100_10/',
        'models/mnist_bnn_4_blk_784_50_50_50_50_10/',
        ]),
    ('--argmax', [
        'bnn_encoding/argmax/output_variable_01.lp',
        ]),
    ],

'hamming-big/100': [
    ('--hamming-distance', ['4', '3', '2', '1', '0']),
    ('-t', ['8']*4),
    ('-T', ['600']),
    ('--input-base', [
        'inputs/instance_0_100.txt',
        ]),
    ('--model', [
        'models/mnist_bnn_1_blk_100_100_10/',
        'models/mnist_bnn_1_blk_100_50_10/',
        'models/mnist_bnn_2_blk_100_100_50_10/',
        'models/mnist_bnn_2_blk_100_50_20_10/',
        ]),
    ('--argmax', [
        'bnn_encoding/argmax/output_variable_01.lp',
        ]),
    ],

'fixed-big/25': [
    ('--fixed-bits', [
        'fixed_bits/fixed_0_25.txt',
        'fixed_bits/fixed_4_25.txt',
        'fixed_bits/fixed_8_25.txt',
        'fixed_bits/fixed_12_25.txt',
        'fixed_bits/fixed_16_25.txt',
        'fixed_bits/fixed_20_25.txt',
        'fixed_bits/fixed_24_25.txt',
        ]),
    ('--fixed-bits-encoding', [
        'bnn_encoding/input_region/fixed_input.lp',
        ]),
    ('-t', ['8']*4),
    ('-T', ['600']),
    ('--input-base', [
        'inputs/instance_0_25.txt',
        ]),
    ('--model', [
        'models/mnist_bnn_3_blk_25_25_25_20_10/',
        ]),
    ('--argmax', [
        'bnn_encoding/argmax/output_variable_01.lp',
        ]),
    ],

'fixed-big/36': [
    ('--fixed-bits', [
        'fixed_bits/fixed_0_36.txt',
        'fixed_bits/fixed_4_36.txt',
        'fixed_bits/fixed_8_36.txt',
        'fixed_bits/fixed_12_36.txt',
        'fixed_bits/fixed_16_36.txt',
        'fixed_bits/fixed_20_36.txt',
        'fixed_bits/fixed_24_36.txt',
        ]),
    ('--fixed-bits-encoding', [
        'bnn_encoding/input_region/fixed_input.lp',
        ]),
    ('-t', ['8']*4),
    ('-T', ['600']),
    ('--input-base', [
        'inputs/instance_0_36.txt',
        ]),
    ('--model', [
        'models/mnist_bnn_2_blk_36_15_10_10/',
        ]),
    ('--argmax', [
        'bnn_encoding/argmax/output_variable_01.lp',
        ]),
    ],

'fixed-big/16': [
    ('--fixed-bits', [
        'fixed_bits/fixed_0_16.txt',
        'fixed_bits/fixed_4_16.txt',
        'fixed_bits/fixed_8_16.txt',
        'fixed_bits/fixed_12_16.txt',
        'fixed_bits/fixed_16_16.txt',
        ]),
    ('--fixed-bits-encoding', [
        'bnn_encoding/input_region/fixed_input.lp',
        ]),
    ('-t', ['8']*4),
    ('-T', ['600']),
    ('--input-base', [
        'inputs/instance_0_16.txt',
        ]),
    ('--model', [
        'models/mnist_bnn_2_blk_16_25_20_10/',
        'models/mnist_bnn_3_blk_16_64_32_20_10/',
        ]),
    ('--argmax', [
        'bnn_encoding/argmax/output_variable_01.lp',
        ]),
    ],

'fixed-big/64': [
    ('--fixed-bits', [
        'fixed_bits/fixed_0_64.txt',
        'fixed_bits/fixed_4_64.txt',
        'fixed_bits/fixed_8_64.txt',
        'fixed_bits/fixed_12_64.txt',
        'fixed_bits/fixed_16_64.txt',
        'fixed_bits/fixed_20_64.txt',
        'fixed_bits/fixed_24_64.txt',
        ]),
    ('--fixed-bits-encoding', [
        'bnn_encoding/input_region/fixed_input.lp',
        ]),
    ('-t', ['8']*4),
    ('-T', ['600']),
    ('--input-base', [
        'inputs/instance_0_64.txt',
        ]),
    ('--model', [
        'models/mnist_bnn_1_blk_64_10_10/',
        ]),
    ('--argmax', [
        'bnn_encoding/argmax/output_variable_01.lp',
        ]),
    ],

'fixed-big/400': [
    ('--fixed-bits', [
        'fixed_bits/fixed_0_400.txt',
        'fixed_bits/fixed_4_400.txt',
        'fixed_bits/fixed_8_400.txt',
        'fixed_bits/fixed_12_400.txt',
        'fixed_bits/fixed_16_400.txt',
        'fixed_bits/fixed_20_400.txt',
        'fixed_bits/fixed_24_400.txt',
        ]),
    ('--fixed-bits-encoding', [
        'bnn_encoding/input_region/fixed_input.lp',
        ]),
    ('-t', ['8']*4),
    ('-T', ['600']),
    ('--input-base', [
        'inputs/instance_0_400.txt',
        ]),
    ('--model', [
        'models/mnist_bnn_1_blk_400_100_10/',
        ]),
    ('--argmax', [
        'bnn_encoding/argmax/output_variable_01.lp',
        ]),
    ],

'fixed-big/784': [
    ('--fixed-bits', [
        'fixed_bits/fixed_0_784.txt',
        'fixed_bits/fixed_4_784.txt',
        'fixed_bits/fixed_8_784.txt',
        'fixed_bits/fixed_12_784.txt',
        'fixed_bits/fixed_16_784.txt',
        'fixed_bits/fixed_20_784.txt',
        'fixed_bits/fixed_24_784.txt',
        ]),
    ('--fixed-bits-encoding', [
        'bnn_encoding/input_region/fixed_input.lp',
        ]),
    ('-t', ['8']*4),
    ('-T', ['600']),
    ('--input-base', [
        'inputs/instance_0_784.txt',
        ]),
    ('--model', [
        'models/mnist_bnn_1_blk_784_100_10/',
        'models/mnist_bnn_4_blk_784_50_50_50_50_10/',
        ]),
    ('--argmax', [
        'bnn_encoding/argmax/output_variable_01.lp',
        ]),
    ],

'fixed-big/100': [
    ('--fixed-bits', [
        'fixed_bits/fixed_0_100.txt',
        'fixed_bits/fixed_4_100.txt',
        'fixed_bits/fixed_8_100.txt',
        'fixed_bits/fixed_12_100.txt',
        'fixed_bits/fixed_16_100.txt',
        'fixed_bits/fixed_20_100.txt',
        'fixed_bits/fixed_24_100.txt',
        ]),
    ('--fixed-bits-encoding', [
        'bnn_encoding/input_region/fixed_input.lp',
        ]),
    ('-t', ['8']*4),
    ('-T', ['600']),
    ('--input-base', [
        'inputs/instance_0_100.txt',
        ]),
    ('--model', [
        'models/mnist_bnn_1_blk_100_100_10/',
        'models/mnist_bnn_1_blk_100_50_10/',
        'models/mnist_bnn_2_blk_100_100_50_10/',
        'models/mnist_bnn_2_blk_100_50_20_10/',
        ]),
    ('--argmax', [
        'bnn_encoding/argmax/output_variable_01.lp',
        ]),
    ],
}
