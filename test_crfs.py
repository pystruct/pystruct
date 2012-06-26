import numpy as np
from numpy.testing import assert_array_equal

import toy_datasets
from crf import BinaryGridCRF, MultinomialGridCRF
from pyqpbo import binary_grid, alpha_expansion_grid

# why have binary and multinomial different numbers of parameters?


def test_binary_grid():
    # test handling on unaries for binary grid CRFs
    for ds in toy_datasets.binary:
        X, Y = ds(n_samples=1)
        x, y = X[0], Y[0]
        crf = BinaryGridCRF()
        w_unaries_only = np.zeros(4)
        w_unaries_only[0] = 1.
        # test that inference with unaries only is the
        # same as argmax
        inf_unaries = crf.inference(x, w_unaries_only)

        pw_z = np.zeros((2, 2), dtype=np.int32)
        un = np.ascontiguousarray(
                -1000 * x).astype(np.int32)
        unaries = binary_grid(un, pw_z)
        assert_array_equal(inf_unaries, unaries)
        assert_array_equal(inf_unaries, np.argmax(x, axis=2))


def test_multinomial_grid_binary():
    # test handling on unaries for multinomial grid CRFs
    # on binary datasets
    for ds in toy_datasets.binary:
        X, Y = ds(n_samples=1)
        x, y = X[0], Y[0]
        crf = MultinomialGridCRF()
        w_unaries_only = np.zeros(5)
        w_unaries_only[:2] = 1.
        # test that inference with unaries only is the
        # same as argmax
        inf_unaries = crf.inference(x, w_unaries_only)

        pw_z = np.zeros((2, 2), dtype=np.int32)
        un = np.ascontiguousarray(
                -1000 * x).astype(np.int32)
        unaries = binary_grid(un, pw_z)
        assert_array_equal(inf_unaries, unaries)
        assert_array_equal(inf_unaries, np.argmax(x, axis=2))


def test_multinomial_grid():
    # test handling on unaries for multinomial grid CRFs
    # on binary datasets
    for ds in toy_datasets.multinomial:
        X, Y = ds(n_samples=1)
        x, y = X[0], Y[0]
        n_labels = len(np.unique(Y))
        crf = MultinomialGridCRF(n_states=n_labels)
        w_unaries_only = np.zeros(crf.size_psi)
        w_unaries_only[:n_labels] = 1.
        # test that inference with unaries only is the
        # same as argmax
        inf_unaries = crf.inference(x, w_unaries_only)

        pw_z = np.zeros((n_labels, n_labels), dtype=np.int32)
        un = np.ascontiguousarray(
                -1000 * x).astype(np.int32)
        unaries = alpha_expansion_grid(un, pw_z)
        assert_array_equal(inf_unaries, unaries)
        assert_array_equal(inf_unaries, np.argmax(x, axis=2))
