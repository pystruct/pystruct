import itertools

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_equal, assert_almost_equal, assert_raises

import pystruct.toy_datasets as toy
from pystruct.problems import GridCRF
from pystruct.inference import inference_lp
from pystruct.utils import make_grid_edges, find_constraint, compute_energy


from IPython.core.debugger import Tracer
tracer = Tracer()


def test_continuous_y():
    for inference_method in ["lp", "ad3"]:
        X, Y = toy.generate_blocks(n_samples=1)
        x, y = X[0], Y[0]
        w = np.array([1, 1,
                      0,
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
    while not found_fractional:
        x = np.random.normal(size=(2, 2, 3))
        unary_params = np.ones(3)
        pairwise_params = np.random.normal() * np.eye(3)
        edges = make_grid_edges(x)
        # check map inference
        inf_res, energy_lp = inference_lp(x * unary_params, pairwise_params,
                                          edges=edges, relaxed=True,
                                          return_energy=True, exact=True)
        found_fractional = np.any(np.max(inf_res[0], axis=-1) != 1)
        energy_svm = compute_energy(x, inf_res, unary_params,
                                    pairwise_params, neighborhood=4)

        assert_almost_equal(energy_lp, -energy_svm)


def test_loss_augmentation():
    X, Y = toy.generate_blocks(n_samples=1)
    x, y = X[0], Y[0]
    w = np.array([1., 1.,
                  0.,
                  -4., 0.])
    unary_params = w[:2]
    pairwise_flat = np.asarray(w[2:])
    pairwise_params = np.zeros((2, 2))
    pairwise_params[np.tri(2, dtype=np.bool)] = pairwise_flat
    pairwise_params = (pairwise_params + pairwise_params.T
                       - np.diag(np.diag(pairwise_params)))
    crf = GridCRF()
    x_loss_augmented = crf.loss_augment(x, y, w)
    y_hat = crf.inference(x_loss_augmented, w)
    # test that loss_augmented_inference does the same
    y_hat_2 = crf.loss_augmented_inference(x, y, w)
    assert_array_equal(y_hat_2, y_hat)
    energy = compute_energy(x, y_hat, unary_params, pairwise_params)
    energy_loss_augmented = compute_energy(x_loss_augmented, y_hat,
                                           unary_params, pairwise_params)

    assert_almost_equal(energy + crf.loss(y, y_hat), energy_loss_augmented)

    # with zero in w:
    unary_params[1] = 0
    assert_raises(ValueError, crf.loss_augment, x, y, w)


def test_binary_blocks_crf():
    X, Y = toy.generate_blocks(n_samples=1)
    x, y = X[0], Y[0]
    w = np.array([1, 1,
                  0,
                  -4, 0])
    for inference_method in ['dai', 'qpbo', 'lp', 'ad3']:
        crf = GridCRF(inference_method=inference_method)
        y_hat = crf.inference(x, w)
        assert_array_equal(y, y_hat)


def test_binary_blocks_crf_n8_lp():
    X, Y = toy.generate_blocks(n_samples=1, noise=1)
    x, y = X[0], Y[0]
    w = np.array([1, 1,
                  1,
                  -1.4, 1])
    crf = GridCRF(inference_method="lp", neighborhood=8)
    y_hat = crf.inference(x, w)
    assert_array_equal(y, y_hat)


def test_blocks_multinomial_crf():
    X, Y = toy.generate_blocks_multinomial(n_samples=1)
    x, y = X[0], Y[0]
    w = np.array([1., 1., 1.,
                 .4,
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
            w_unaries_only = np.zeros(5)
            w_unaries_only[:2] = 1.
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
            w_unaries_only[:n_labels] = 1.
            # test that inference with unaries only is the
            # same as argmax
            inf_unaries = crf.inference(x, w_unaries_only)

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
