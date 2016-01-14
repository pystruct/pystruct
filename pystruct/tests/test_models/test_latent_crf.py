import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_equal, assert_almost_equal)

from pystruct.datasets import (generate_big_checker,
                               generate_blocks_multinomial, generate_blocks)
from pystruct.utils import (exhaustive_loss_augmented_inference,
                            make_grid_edges, find_constraint)
from pystruct.models import (LatentGridCRF, LatentDirectionalGridCRF,
                             LatentGraphCRF)
from pystruct.models.latent_grid_crf import kmeans_init
from pystruct.inference import get_installed


def test_k_means_initialization():
    n_samples = 10
    X, Y = generate_big_checker(n_samples=n_samples)
    edges = [make_grid_edges(x, return_lists=True) for x in X]
    # flatten the grid
    Y = Y.reshape(Y.shape[0], -1)
    X = X.reshape(X.shape[0], -1, X.shape[-1])
    n_labels = len(np.unique(Y))
    X = X.reshape(n_samples, -1, n_labels)

    # sanity check for one state
    H = kmeans_init(X, Y, edges, n_states_per_label=[1] * n_labels,
                    n_labels=n_labels)
    H = np.vstack(H)
    assert_array_equal(Y, H)

    # check number of states
    H = kmeans_init(X, Y, edges, n_states_per_label=[3] * n_labels,
                    n_labels=n_labels)
    H = np.vstack(H)
    assert_array_equal(np.unique(H), np.arange(6))
    assert_array_equal(Y, H // 3)

    # for dataset with more than two states
    X, Y = generate_blocks_multinomial(n_samples=10)
    edges = [make_grid_edges(x, return_lists=True) for x in X]
    Y = Y.reshape(Y.shape[0], -1)
    X = X.reshape(X.shape[0], -1, X.shape[-1])
    n_labels = len(np.unique(Y))

    # sanity check for one state
    H = kmeans_init(X, Y, edges, n_states_per_label=[1] * n_labels,
                    n_labels=n_labels)
    H = np.vstack(H)
    assert_array_equal(Y, H)

    # check number of states
    H = kmeans_init(X, Y, edges, n_states_per_label=[2] * n_labels,
                    n_labels=n_labels)
    H = np.vstack(H)
    assert_array_equal(np.unique(H), np.arange(6))
    assert_array_equal(Y, H // 2)


def test_k_means_initialization_grid_crf():
    # with only 1 state per label, nothing happends
    X, Y = generate_big_checker(n_samples=10)
    crf = LatentGridCRF(n_states_per_label=1, n_features=2, n_labels=2)
    H = crf.init_latent(X, Y)
    assert_array_equal(Y, H)


def test_k_means_initialization_graph_crf():
    # with only 1 state per label, nothing happends
    X, Y = generate_big_checker(n_samples=10)
    crf = LatentGraphCRF(n_states_per_label=1, n_features=2, n_labels=2)
    # convert grid model to graph model
    X = [(x.reshape(-1, x.shape[-1]), make_grid_edges(x, return_lists=False))
         for x in X]

    H = crf.init_latent(X, Y)
    assert_array_equal(Y, H)


def test_k_means_initialization_directional_grid_crf():
    X, Y = generate_big_checker(n_samples=10)
    crf = LatentDirectionalGridCRF(n_states_per_label=1, n_features=2,
                                   n_labels=2)
    #crf.initialize(X, Y)
    H = crf.init_latent(X, Y)
    assert_array_equal(Y, H)


def test_blocks_crf_unaries():
    X, Y = generate_blocks(n_samples=1)
    x, _ = X[0], Y[0]
    unary_weights = np.repeat(np.eye(2), 2, axis=0)
    pairwise_weights = np.array([0,
                                 0, 0,
                                 0, 0, 0,
                                 0, 0, 0, 0])
    w = np.hstack([unary_weights.ravel(), pairwise_weights])
    crf = LatentGridCRF(n_states_per_label=2, n_labels=2, n_features=2)
    h_hat = crf.inference(x, w)
    assert_array_equal(h_hat // 2, np.argmax(x, axis=-1))


def test_blocks_crf():
    X, Y = generate_blocks(n_samples=1)
    x, y = X[0], Y[0]
    pairwise_weights = np.array([0,
                                 0,  0,
                                -4, -4, 0,
                                -4, -4, 0, 0])
    unary_weights = np.repeat(np.eye(2), 2, axis=0)
    w = np.hstack([unary_weights.ravel(), pairwise_weights])
    crf = LatentGridCRF(n_states_per_label=2, n_labels=2, n_features=2)
    h_hat = crf.inference(x, w)
    assert_array_equal(y, h_hat // 2)

    h = crf.latent(x, y, w)
    assert_equal(crf.loss(h, h_hat), 0)


def test_blocks_crf_directional():
    # test latent directional CRF on blocks
    # test that all results are the same as equivalent LatentGridCRF
    X, Y = generate_blocks(n_samples=1)
    x, y = X[0], Y[0]
    pairwise_weights = np.array([0,
                                 0,  0,
                                -4, -4, 0,
                                -4, -4, 0, 0])
    unary_weights = np.repeat(np.eye(2), 2, axis=0)
    w = np.hstack([unary_weights.ravel(), pairwise_weights])
    pw_directional = np.array([0,   0, -4, -4,
                               0,   0, -4, -4,
                               -4, -4,  0,  0,
                               -4, -4,  0,  0,
                               0,   0, -4, -4,
                               0,   0, -4, -4,
                               -4, -4,  0,  0,
                               -4, -4,  0,  0])
    w_directional = np.hstack([unary_weights.ravel(), pw_directional])
    crf = LatentGridCRF(n_states_per_label=2)
    crf.initialize(X, Y)
    directional_crf = LatentDirectionalGridCRF(n_states_per_label=2)
    directional_crf.initialize(X, Y)
    h_hat = crf.inference(x, w)
    h_hat_d = directional_crf.inference(x, w_directional)
    assert_array_equal(h_hat, h_hat_d)

    h = crf.latent(x, y, w)
    h_d = directional_crf.latent(x, y, w_directional)
    assert_array_equal(h, h_d)

    h_hat = crf.loss_augmented_inference(x, y, w)
    h_hat_d = directional_crf.loss_augmented_inference(x, y, w_directional)
    assert_array_equal(h_hat, h_hat_d)

    joint_feature = crf.joint_feature(x, h_hat)
    joint_feature_d = directional_crf.joint_feature(x, h_hat)
    assert_array_equal(np.dot(joint_feature, w), np.dot(joint_feature_d, w_directional))


def test_latent_consistency_zero_pw_graph():
    crf = LatentGraphCRF(n_labels=2, n_features=2, n_states_per_label=2)
    for i in range(10):
        w = np.zeros(18)
        w[:8] = np.random.normal(size=8)
        y = np.random.randint(2, size=(5))
        x = np.random.normal(size=(5, 2))
        h = crf.latent((x, np.zeros((0, 2), dtype=np.int)), y, w)
        assert_array_equal(h // 2, y)


def test_latent_consistency_graph():
    crf = LatentGraphCRF(n_labels=2, n_features=2, n_states_per_label=2)
    for i in range(10):
        w = np.random.normal(size=18)
        y = np.random.randint(2, size=(4))
        x = np.random.normal(size=(4, 2))
        e = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int)
        h = crf.latent((x, e), y, w)
        assert_array_equal(h // 2, y)


def test_loss_augmented_inference_energy_graph():
    crf = LatentGraphCRF(n_labels=2, n_features=2, n_states_per_label=2)
    for i in range(10):
        w = np.random.normal(size=18)
        y = np.random.randint(2, size=(3))
        x = np.random.normal(size=(3, 2))
        e = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int)
        h_hat, energy = crf.loss_augmented_inference((x, e), y * 2, w,
                                                     relaxed=True,
                                                     return_energy=True)
        assert_almost_equal(-energy, np.dot(w, crf.joint_feature((x, e), h_hat))
                            + crf.loss(y * 2, h_hat))


def test_latent_consistency_zero_pw_grid():
    crf = LatentGridCRF(n_labels=2, n_features=2, n_states_per_label=2)
    for i in range(10):
        w = np.zeros(18)
        w[:8] = np.random.normal(size=8)
        y = np.random.randint(2, size=(5, 5))
        x = np.random.normal(size=(5, 5, 2))
        h = crf.latent(x, y, w)
        assert_array_equal(h // 2, y)


def test_latent_consistency_grid():
    crf = LatentGridCRF(n_labels=2, n_features=2, n_states_per_label=2)
    for i in range(10):
        w = np.random.normal(size=18)
        y = np.random.randint(2, size=(4, 4))
        x = np.random.normal(size=(4, 4, 2))
        h = crf.latent(x, y, w)
        assert_array_equal(h // 2, y)


def test_loss_augmented_inference_exhaustive_grid():
    crf = LatentGridCRF(n_labels=2, n_features=2, n_states_per_label=2)
    for i in range(10):
        w = np.random.normal(size=18)
        y = np.random.randint(2, size=(2, 2))
        x = np.random.normal(size=(2, 2, 2))
        h_hat = crf.loss_augmented_inference(x, y * 2, w)
        h = exhaustive_loss_augmented_inference(crf, x, y * 2, w)
        assert_array_equal(h, h_hat)


def test_continuous_y():
    for inference_method in get_installed(["lp", "ad3"]):
        X, Y = generate_blocks(n_samples=1)
        x, y = X[0], Y[0]
        w = np.array([1, 0,  # unary
                      0, 1,
                      0,     # pairwise
                      -4, 0])

        crf = LatentGridCRF(n_labels=2, n_features=2, n_states_per_label=1,
                            inference_method=inference_method)
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
        assert_array_almost_equal(joint_feature, joint_feature_cont, 4)

        const = find_constraint(crf, x, y, w, relaxed=False)
        const_cont = find_constraint(crf, x, y, w, relaxed=True)

        # djoint_feature and loss are equal:
        assert_array_almost_equal(const[1], const_cont[1], 4)
        assert_almost_equal(const[2], const_cont[2], 4)

        if isinstance(const_cont[0], tuple):
            # returned y_hat is one-hot version of other
            assert_array_equal(const[0], np.argmax(const_cont[0][0], axis=-1))

            # test loss:
            assert_almost_equal(crf.loss(y, const[0]),
                                crf.continuous_loss(y, const_cont[0][0]), 4)
