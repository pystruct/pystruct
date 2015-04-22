import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)

from pystruct.datasets import (generate_blocks, generate_blocks_multinomial,
                               binary, multinomial)
from pystruct.models import GridCRF
from pystruct.utils import (find_constraint, exhaustive_inference,
                            exhaustive_loss_augmented_inference)
from pystruct.inference import get_installed


def test_continuous_y():
    for inference_method in get_installed(["lp", "ad3"]):
        X, Y = generate_blocks(n_samples=1)
        x, y = X[0], Y[0]
        w = np.array([1, 0,  # unary
                      0, 1,
                      0,     # pairwise
                      -4, 0])

        crf = GridCRF(inference_method=inference_method)
        crf.initialize(X, Y)
        joint_feature = crf.joint_feature(x, y)
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

        joint_feature_cont = crf.joint_feature(x, (y_cont, pw))
        assert_array_almost_equal(joint_feature, joint_feature_cont)

        const = find_constraint(crf, x, y, w, relaxed=False)
        const_cont = find_constraint(crf, x, y, w, relaxed=True)

        # djoint_feature and loss are equal:
        assert_array_almost_equal(const[1], const_cont[1], 4)
        assert_almost_equal(const[2], const_cont[2], 4)

        # returned y_hat is one-hot version of other
        if isinstance(const_cont[0], tuple):
            assert_array_equal(const[0], np.argmax(const_cont[0][0], axis=-1))

            # test loss:
            assert_almost_equal(crf.loss(y, const[0]),
                                crf.continuous_loss(y, const_cont[0][0]), 4)


def test_energy_lp():
    # make sure that energy as computed by ssvm is the same as by lp
    np.random.seed(0)
    found_fractional = False
    for inference_method in get_installed(["lp", "ad3"]):
        crf = GridCRF(n_states=3, n_features=4,
                      inference_method=inference_method)
        while not found_fractional:
            x = np.random.normal(size=(2, 2, 4))
            w = np.random.uniform(size=crf.size_joint_feature)
            inf_res, energy_lp = crf.inference(x, w, relaxed=True,
                                               return_energy=True)
            assert_almost_equal(energy_lp,
                                -np.dot(w, crf.joint_feature(x, inf_res)))
            found_fractional = np.any(np.max(inf_res[0], axis=-1) != 1)


def test_loss_augmentation():
    X, Y = generate_blocks(n_samples=1)
    x, y = X[0], Y[0]
    w = np.array([1, 0,  # unary
                  0, 1,
                  0,     # pairwise
                  -4, 0])
    crf = GridCRF()
    crf.initialize(X, Y)
    y_hat, energy = crf.loss_augmented_inference(x, y, w, return_energy=True)

    assert_almost_equal(energy + crf.loss(y, y_hat),
                        -np.dot(w, crf.joint_feature(x, y_hat)))


def test_binary_blocks_crf():
    X, Y = generate_blocks(n_samples=1)
    x, y = X[0], Y[0]
    w = np.array([1, 0,  # unary
                  0, 1,
                  0,     # pairwise
                  -4, 0])
    for inference_method in get_installed(['qpbo', 'lp', 'ad3', 'ogm']):
        crf = GridCRF(inference_method=inference_method)
        crf.initialize(X, Y)
        y_hat = crf.inference(x, w)
        assert_array_equal(y, y_hat)


def test_binary_blocks_crf_n8_lp():
    X, Y = generate_blocks(n_samples=1, noise=1)
    x, y = X[0], Y[0]
    w = np.array([1, 0,  # unary
                  0, 1,
                  1,     # pairwise
                  -1.4, 1])
    crf = GridCRF(neighborhood=8)
    crf.initialize(X, Y)
    y_hat = crf.inference(x, w)
    assert_array_equal(y, y_hat)


def test_blocks_multinomial_crf():
    X, Y = generate_blocks_multinomial(n_samples=1, size_x=9, seed=0)
    x, y = X[0], Y[0]
    w = np.array([1., 0., 0.,  # unaryA
                  0., 1., 0.,
                  0., 0., 1.,
                 .4,           # pairwise
                 -.3, .3,
                 -.5, -.1, .3])
    for inference_method in get_installed():
        crf = GridCRF(inference_method=inference_method)
        crf.initialize(X, Y)
        y_hat = crf.inference(x, w)
        assert_array_equal(y, y_hat)


def test_binary_grid_unaries():
    # test handling on unaries for binary grid CRFs
    for ds in binary:
        X, Y = ds(n_samples=1)
        x, y = X[0], Y[0]
        for inference_method in get_installed():
            crf = GridCRF(inference_method=inference_method)
            crf.initialize(X, Y)
            w_unaries_only = np.zeros(7)
            w_unaries_only[:4] = np.eye(2).ravel()
            # test that inference with unaries only is the
            # same as argmax
            inf_unaries = crf.inference(x, w_unaries_only)

            assert_array_equal(inf_unaries, np.argmax(x, axis=2),
                               "Wrong unary inference for %s"
                               % inference_method)
            assert(np.mean(inf_unaries == y) > 0.5)

            # check that the right thing happens on noise-free data
            X, Y = ds(n_samples=1, noise=0)
            inf_unaries = crf.inference(X[0], w_unaries_only)
            assert_array_equal(inf_unaries, Y[0],
                               "Wrong unary result for %s"
                               % inference_method)


def test_multinomial_grid_unaries():
    # test handling on unaries for multinomial grid CRFs
    # on multinomial datasets
    for ds in multinomial:
        X, Y = ds(n_samples=1, size_x=9)
        x = X[0]
        n_labels = len(np.unique(Y))
        crf = GridCRF(n_states=n_labels)
        crf.initialize(X, Y)
        w_unaries_only = np.zeros(crf.size_joint_feature)
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
    # tests qpbo inference against brute force
    # on random data / weights
    np.random.seed(0)
    for i in range(10):
        x = np.random.uniform(-1, 1, size=(3, 2))
        x = np.dstack([-x, np.zeros_like(x)]).copy()
        crf = GridCRF(n_features=2, n_states=2)
        w = np.random.uniform(-1, 1, size=7)
        # check map inference
        y_hat = crf.inference(x, w)
        y_ex = exhaustive_inference(crf, x, w)
        assert_array_equal(y_hat, y_ex)


def test_binary_crf_exhaustive_loss_augmented():
    # tests qpbo inference against brute force
    # on random data / weights
    np.random.seed(0)
    for inference_method in get_installed(['qpbo', 'lp']):
        crf = GridCRF(n_states=2, n_features=2,
                      inference_method=inference_method)
        for i in range(10):
            # generate data and weights
            y = np.random.randint(2, size=(3, 2))
            x = np.random.uniform(-1, 1, size=(3, 2))
            x = np.dstack([-x, np.zeros_like(x)])
            w = np.random.uniform(-1, 1, size=7)
            # check loss augmented map inference
            y_hat = crf.loss_augmented_inference(x, y, w)
            y_ex = exhaustive_loss_augmented_inference(crf, x, y, w)
            assert_array_equal(y_hat, y_ex)
