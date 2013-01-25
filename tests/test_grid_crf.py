import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_equal, assert_almost_equal

import pystruct.toy_datasets as toy
from pystruct.problems import GridCRF
from pystruct.utils import (find_constraint, exhaustive_inference,
                            exhaustive_loss_augmented_inference)


from IPython.core.debugger import Tracer
tracer = Tracer()


def test_continuous_y():
    for inference_method in ["lp", "ad3"]:
        X, Y = toy.generate_blocks(n_samples=1)
        x, y = X[0], Y[0]
        w = np.array([1, 0,  # unary
                      0, 1,
                      0,     # pairwise
                      -4, 0])

        crf = GridCRF(inference_method=inference_method)
        psi = crf.psi(x, y)
        y_cont = np.zeros_like(x)
        gx, gy = np.indices(x.shape[:-1])
        y_cont[gx, gy, y] = 1
        # need to generate edge marginals
        vert = np.dot(y_cont[1:, :, :].reshape(-1, 2).T,
                      y_cont[:-1, :, :].reshape(-1, 2))
        # horizontal edges
        horz = np.dot(y_cont[:, 1:, :].reshape(-1, 2).T,
                      y_cont[:, :-1, :].reshape(-1, 2))
        pw = vert + horz

        psi_cont = crf.psi(x, (y_cont, pw))
        assert_array_almost_equal(psi, psi_cont)

        const = find_constraint(crf, x, y, w, relaxed=False)
        const_cont = find_constraint(crf, x, y, w, relaxed=True)

        # dpsi and loss are equal:
        assert_array_almost_equal(const[1], const_cont[1])
        assert_almost_equal(const[2], const_cont[2])

        # returned y_hat is one-hot version of other
        assert_array_equal(const[0], np.argmax(const_cont[0][0], axis=-1))

        # test loss:
        assert_equal(crf.loss(y, const[0]),
                     crf.continuous_loss(y, const_cont[0][0]))


def test_energy_lp():
    # make sure that energy as computed by ssvm is the same as by lp
    np.random.seed(0)
    found_fractional = False
    for inference_method in ["lp", "ad3"]:
        crf = GridCRF(n_states=3, n_features=4, inference_method='lp')
        while not found_fractional:
            x = np.random.normal(size=(2, 2, 4))
            w = np.random.uniform(size=crf.size_psi)
            inf_res, energy_lp = crf.inference(x, w, relaxed=True,
                                               return_energy=True)
            assert_almost_equal(energy_lp,
                                -np.dot(w, crf.psi(x, inf_res)))
            found_fractional = np.any(np.max(inf_res[0], axis=-1) != 1)


def test_loss_augmentation():
    X, Y = toy.generate_blocks(n_samples=1)
    x, y = X[0], Y[0]
    w = np.array([1, 0,  # unary
                  0, 1,
                  0,     # pairwise
                  -4, 0])
    crf = GridCRF(inference_method='lp')
    y_hat, energy = crf.loss_augmented_inference(x, y, w, return_energy=True)

    assert_almost_equal(energy + crf.loss(y, y_hat),
                        -np.dot(w, crf.psi(x, y_hat)))


def test_binary_blocks_crf():
    X, Y = toy.generate_blocks(n_samples=1)
    x, y = X[0], Y[0]
    w = np.array([1, 0,  # unary
                  0, 1,
                  0,     # pairwise
                  -4, 0])
    for inference_method in ['dai', 'qpbo', 'lp', 'ad3']:
        crf = GridCRF(inference_method=inference_method)
        y_hat = crf.inference(x, w)
        assert_array_equal(y, y_hat)


def test_binary_blocks_crf_n8_lp():
    X, Y = toy.generate_blocks(n_samples=1, noise=1)
    x, y = X[0], Y[0]
    w = np.array([1, 0,  # unary
                  0, 1,
                  1,     # pairwise
                  -1.4, 1])
    crf = GridCRF(inference_method="lp", neighborhood=8)
    y_hat = crf.inference(x, w)
    assert_array_equal(y, y_hat)


def test_blocks_multinomial_crf():
    X, Y = toy.generate_blocks_multinomial(n_samples=1)
    x, y = X[0], Y[0]
    w = np.array([1., 0., 0.,  # unaryA
                  0., 1., 0.,
                  0., 0., 1.,
                 .4,           # pairwise
                 -.3, .3,
                 -.5, -.1, .3])
    for inference_method in ['dai', 'qpbo', 'lp', 'ad3']:
        crf = GridCRF(n_states=3, inference_method=inference_method)
        y_hat = crf.inference(x, w)
        assert_array_equal(y, y_hat)


def test_binary_grid_unaries():
    # test handling on unaries for binary grid CRFs
    for ds in toy.binary:
        X, Y = ds(n_samples=1)
        x, y = X[0], Y[0]
        for inference_method in ['qpbo', 'lp', 'ad3']:  # dai is to expensive
            crf = GridCRF(inference_method=inference_method)
            w_unaries_only = np.zeros(7)
            w_unaries_only[:4] = np.eye(2).ravel()
            # test that inference with unaries only is the
            # same as argmax
            inf_unaries = crf.inference(x, w_unaries_only)

            assert_array_equal(inf_unaries, np.argmax(x, axis=2),
                               "Wrong unary inference for %s"
                               % inference_method)
            try:
                assert(np.mean(inf_unaries == y) > 0.5)
            except:
                print(ds)

            # check that the right thing happens on noise-free data
            X, Y = ds(n_samples=1, noise=0)
            inf_unaries = crf.inference(X[0], w_unaries_only)
            assert_array_equal(inf_unaries, Y[0],
                               "Wrong unary result for %s"
                               % inference_method)


def test_multinomial_grid_unaries():
    # test handling on unaries for multinomial grid CRFs
    # on multinomial datasets
    for ds in toy.multinomial:
        X, Y = ds(n_samples=1)
        x, y = X[0], Y[0]
        n_labels = len(np.unique(Y))
        for inference_method in ['qpbo', 'lp', 'ad3']:  # dai is to expensive
            crf = GridCRF(n_states=n_labels, inference_method=inference_method)
            w_unaries_only = np.zeros(crf.size_psi)
            w_unaries_only[:n_labels ** 2] = np.eye(n_labels).ravel()
            # test that inference with unaries only is the
            # same as argmax
            inf_unaries = crf.inference(x, w_unaries_only)

            assert_array_equal(inf_unaries, np.argmax(x, axis=2))
            # check that the right thing happens on noise-free data
            X, Y = ds(n_samples=1, noise=0)
            inf_unaries = crf.inference(X[0], w_unaries_only)
            assert_array_equal(inf_unaries, Y[0])


def test_binary_crf_exhaustive():
    # tests graph cut inference against brute force
    # on random data / weights
    np.random.seed(0)
    for i in xrange(50):
        x = np.random.uniform(-1, 1, size=(3, 3))
        x = np.dstack([-x, np.zeros_like(x)]).copy()
        crf = GridCRF()
        w = np.random.uniform(-1, 1, size=7)
        # check map inference
        y_hat = crf.inference(x, w)
        y_ex = exhaustive_inference(crf, x, w)
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
        w = np.random.uniform(-1, 1, size=7)
        crf = GridCRF()
        # check loss augmented map inference
        y_hat = crf.loss_augmented_inference(x, y, w)
        y_ex = exhaustive_loss_augmented_inference(crf, x, y, w)
        #print(y_hat)
        #print(y_ex)
        #print("++++++++++++++++++++++")
        assert_array_equal(y_hat, y_ex)

if __name__ == "__main__":
    test_binary_crf_exhaustive()
    test_binary_crf_exhaustive_loss_augmented()
    test_binary_grid_unaries()
    test_multinomial_grid_unaries()
