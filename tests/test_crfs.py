import numpy as np
from numpy.testing import assert_array_equal

import toy_datasets as toy
from crf import GridCRF
from pyqpbo import binary_grid, alpha_expansion_grid

import itertools

from IPython.core.debugger import Tracer
tracer = Tracer()
# why have binary and multinomial different numbers of parameters?


def test_binary_blocks_crf():
    X, Y = toy.generate_blocks(n_samples=1)
    x, y = X[0], Y[0]
    w = np.array([1, 1,
                  0,
                  -4, 0])
    crf = GridCRF()
    y_hat = crf.inference(x, w)
    assert_array_equal(y, y_hat)


def test_blocks_multinomial_crf():
    X, Y = toy.generate_blocks_multinomial(n_samples=1)
    x, y = X[0], Y[0]
    w = np.array([1., 1., 1.,
                 .4,
                 -.3, .3,
                 -.5, -.1, .3])
    crf = GridCRF(n_states=3)
    y_hat = crf.inference(x, w)
    assert_array_equal(y, y_hat)


def test_binary_grid_unaries():
    # test handling on unaries for binary grid CRFs
    for ds in toy.binary:
        X, Y = ds(n_samples=1)
        x, y = X[0], Y[0]
        crf = GridCRF()
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
        try:
            assert(np.mean(inf_unaries == y) > 0.5)
        except:
            print(ds)

        # check that the right thing happens on noise-free data
        X, Y = ds(n_samples=1, noise=0)
        inf_unaries = crf.inference(X[0], w_unaries_only)
        assert_array_equal(inf_unaries, Y[0])


def test_multinomial_grid_unaries():
    # test handling on unaries for multinomial grid CRFs
    # on multinomial datasets
    for ds in toy.multinomial:
        X, Y = ds(n_samples=1)
        x, y = X[0], Y[0]
        n_labels = len(np.unique(Y))
        crf = GridCRF(n_states=n_labels)
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
        # check that the right thing happens on noise-free data
        X, Y = ds(n_samples=1, noise=0)
        inf_unaries = crf.inference(X[0], w_unaries_only)
        assert_array_equal(inf_unaries, Y[0])


def exhausive_inference_binary(problem, x, w):
    size = np.prod(x.shape[:-1])
    best_y = None
    best_energy = np.inf
    for y_hat in itertools.product([0, 1], repeat=size):
        y_hat = np.array(y_hat).reshape(x.shape[:-1])
        psi = problem.psi(x, y_hat)
        energy = -np.dot(w, psi)
        if energy < best_energy:
            best_energy = energy
            best_y = y_hat
    return best_y


def exhausive_loss_augmented_inference_binary(problem, x, y, w):
    size = np.prod(x.shape[:-1])
    best_y = None
    best_energy = np.inf
    for y_hat in itertools.product([0, 1], repeat=size):
        y_hat = np.array(y_hat).reshape(x.shape[:-1])
        psi = problem.psi(x, y_hat)
        energy = -problem.loss(y, y_hat) - np.dot(w, psi)
        if energy < best_energy:
            best_energy = energy
            best_y = y_hat
    return best_y


def test_binary_crf_exhaustive():
    # tests graph cut inference against brute force
    # on random data / weights
    np.random.seed(0)
    for i in xrange(50):
        x = np.random.uniform(-1, 1, size=(3, 3))
        x = np.dstack([-x, np.zeros_like(x)]).copy()
        crf = GridCRF()
        w = np.random.uniform(-1, 1, size=5)
        # check map inference
        y_hat = crf.inference(x, w)
        y_ex = exhausive_inference_binary(crf, x, w)
        #print(y_hat)
        #print(y_ex)
        #print("++++++++++++++++++++++")
        assert_array_equal(y_hat, y_ex)


def test_binary_crf_exhaustive_loss_augmented():
    # tests graph cut inference against brute force
    # on random data / weights
    np.random.seed(0)
    for i in xrange(50):
        # generate data and weights
        y = np.random.randint(2, size=(3, 3))
        x = np.random.uniform(-1, 1, size=(3, 3))
        x = np.dstack([-x, np.zeros_like(x)])
        w = np.random.uniform(-1, 1, size=5)
        crf = GridCRF()
        # check loss augmented map inference
        y_hat = crf.loss_augmented_inference(x, y, w)
        y_ex = exhausive_loss_augmented_inference_binary(crf, x, y, w)
        #print(y_hat)
        #print(y_ex)
        #print("++++++++++++++++++++++")
        assert_array_equal(y_hat, y_ex)

if __name__ == "__main__":
    test_binary_crf_exhaustive()
    test_binary_crf_exhaustive_loss_augmented()
    test_binary_grid_unaries()
    test_multinomial_grid_unaries()
