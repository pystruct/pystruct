import numpy as np
from numpy.testing import assert_array_equal

from pystruct.inference import inference_dai, inference_ad3, inference_lp


def test_chain():
    # test LP, AD3, AD3-BB and JT on a chain.
    # they should all be exact
    rnd = np.random.RandomState(0)
    for i in xrange(10):
        forward = np.c_[np.arange(9), np.arange(1, 10)]
        backward = np.c_[np.arange(1, 10), np.arange(9)]
        unary_potentials = rnd.normal(size=(10, 3))
        pairwise_potentials = rnd.normal(size=(3, 3))
        # test that reversing edges is same as transposing pairwise potentials
        y_ad3_forward = inference_ad3(unary_potentials, pairwise_potentials,
                                      forward)
        y_ad3_backward = inference_ad3(unary_potentials, pairwise_potentials.T,
                                       backward)
        assert_array_equal(y_ad3_forward, y_ad3_backward)
        for chain in [forward, backward]:
            y_dai = inference_dai(unary_potentials, pairwise_potentials, chain,
                                  alg='jt')
            y_ad3 = inference_ad3(unary_potentials, pairwise_potentials, chain)
            y_ad3bb = inference_ad3(unary_potentials, pairwise_potentials,
                                    chain, branch_and_bound=True)
            y_lp = inference_lp(unary_potentials, pairwise_potentials, chain)
            assert_array_equal(y_lp, y_ad3bb)
            assert_array_equal(y_dai, y_ad3)
            assert_array_equal(y_ad3, y_ad3bb)
            print(y_dai)
