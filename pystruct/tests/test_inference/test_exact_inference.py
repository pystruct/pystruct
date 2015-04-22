import numpy as np
from numpy.testing import assert_array_equal

from pystruct.inference import get_installed, inference_dispatch


def test_chain():
    # test LP, AD3, AD3-BB and JT on a chain.
    # they should all be exact
    rnd = np.random.RandomState(0)
    algorithms = get_installed([('ad3', {'branch_and_bound': False}),
                                ('ad3', {'branch_and_bound': True}),
                                ('ogm', {'alg': 'dyn'}),
                                ('ogm', {'alg': 'dd'}),
                                ('ogm', {'alg': 'trw'})])
    n_states = 3
    n_nodes = 10

    for i in range(10):
        forward = np.c_[np.arange(n_nodes - 1), np.arange(1, n_nodes)]
        backward = np.c_[np.arange(1, n_nodes), np.arange(n_nodes - 1)]
        unary_potentials = rnd.normal(size=(n_nodes, n_states))
        pairwise_potentials = rnd.normal(size=(n_states, n_states))
        # test that reversing edges is same as transposing pairwise potentials
        y_forward = inference_dispatch(unary_potentials, pairwise_potentials,
                                       forward, 'lp')
        y_backward = inference_dispatch(unary_potentials,
                                        pairwise_potentials.T, backward, 'lp')
        assert_array_equal(y_forward, y_backward)
        for chain in [forward, backward]:
            y_lp = inference_dispatch(unary_potentials, pairwise_potentials,
                                      chain, 'lp')
            for alg in algorithms:
                if chain is backward and alg[0] == 'ogm':
                    # ogm needs sorted indices
                    continue
                y = inference_dispatch(unary_potentials, pairwise_potentials,
                                       chain, alg)
                assert_array_equal(y, y_lp)
