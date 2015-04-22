import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal, assert_equal)

from nose.tools import assert_raises

from pystruct.models import GraphCRF
from pystruct.inference import get_installed


w = np.array([1, 0,  # unary
              0, 1,
              .22,  # pairwise
              0, .22])

# for directed CRF with non-symmetric weights
w_sym = np.array([1, 0,    # unary
                  0, 1,
                  .22, 0,  # pairwise
                  0, .22])

# triangle
x_1 = np.array([[0, 1], [1, 0], [.4, .6]])
g_1 = np.array([[0, 1], [1, 2], [0, 2]])
# expected result
y_1 = np.array([1, 0, 1])

# chain
x_2 = np.array([[0, 1], [1, 0], [.4, .6]])
g_2 = np.array([[0, 1], [1, 2]])
# expected result
y_2 = np.array([1, 0, 0])


def test_initialize():
    x = (x_1, g_1)
    y = y_1
    crf = GraphCRF(n_states=2, n_features=2)
    # no-op
    crf.initialize([x], [y])

    #test initialization works
    crf = GraphCRF()
    crf.initialize([x], [y])
    assert_equal(crf.n_states, 2)
    assert_equal(crf.n_features, 2)

    crf = GraphCRF(n_states=3)
    assert_raises(ValueError, crf.initialize, X=[x], Y=[y])


def test_graph_crf_inference():
    # create two samples with different graphs
    # two states only, pairwise smoothing
    for inference_method in get_installed(['qpbo', 'lp', 'ad3', 'ogm']):
        crf = GraphCRF(n_states=2, n_features=2,
                       inference_method=inference_method)
        assert_array_equal(crf.inference((x_1, g_1), w), y_1)
        assert_array_equal(crf.inference((x_2, g_2), w), y_2)


def test_directed_graph_crf_inference():
    # create two samples with different graphs
    # two states only, pairwise smoothing
    # same as above, only with full symmetric matrix
    for inference_method in get_installed(['qpbo', 'lp', 'ad3', 'ogm']):
        crf = GraphCRF(n_states=2, n_features=2,
                       inference_method=inference_method, directed=True)
        assert_array_equal(crf.inference((x_1, g_1), w_sym), y_1)
        assert_array_equal(crf.inference((x_2, g_2), w_sym), y_2)


def test_graph_crf_continuous_inference():
    for inference_method in get_installed(['lp', 'ad3']):
        crf = GraphCRF(n_states=2, n_features=2,
                       inference_method=inference_method)
        y_hat = crf.inference((x_1, g_1), w, relaxed=True)
        if isinstance(y_hat, tuple):
            assert_array_equal(np.argmax(y_hat[0], axis=-1), y_1)
        else:
            # ad3 produces integer result if it found the exact solution
            assert_array_equal(y_hat, y_1)

        y_hat = crf.inference((x_2, g_2), w, relaxed=True)
        if isinstance(y_hat, tuple):
            assert_array_equal(np.argmax(y_hat[0], axis=-1), y_2)
        else:
            assert_array_equal(y_hat, y_2)


def test_graph_crf_energy_lp_integral():
    crf = GraphCRF(n_states=2, inference_method='lp', n_features=2)
    inf_res, energy_lp = crf.inference((x_1, g_1), w, relaxed=True,
                                       return_energy=True)
    # integral solution
    assert_array_almost_equal(np.max(inf_res[0], axis=-1), 1)
    y = np.argmax(inf_res[0], axis=-1)
    # energy and joint_feature check out
    assert_almost_equal(energy_lp, -np.dot(w, crf.joint_feature((x_1, g_1), y)), 4)


def test_graph_crf_energy_lp_relaxed():
    crf = GraphCRF(n_states=2, n_features=2)
    for i in range(10):
        w_ = np.random.uniform(size=w.shape)
        inf_res, energy_lp = crf.inference((x_1, g_1), w_, relaxed=True,
                                           return_energy=True)
        assert_almost_equal(energy_lp,
                            -np.dot(w_, crf.joint_feature((x_1, g_1), inf_res)))

    # now with fractional solution
    x = np.array([[0, 0], [0, 0], [0, 0]])
    inf_res, energy_lp = crf.inference((x, g_1), w, relaxed=True,
                                       return_energy=True)
    assert_almost_equal(energy_lp, -np.dot(w, crf.joint_feature((x, g_1), inf_res)))


def test_graph_crf_loss_augment():
    x = (x_1, g_1)
    y = y_1
    crf = GraphCRF()
    crf.initialize([x], [y])
    y_hat, energy = crf.loss_augmented_inference(x, y, w, return_energy=True)
    # check that y_hat fulfills energy + loss condition
    assert_almost_equal(np.dot(w, crf.joint_feature(x, y_hat)) + crf.loss(y, y_hat),
                        -energy)


def test_graph_crf_class_weights():
    # no edges
    crf = GraphCRF(n_states=3, n_features=3)
    w = np.array([1, 0, 0,  # unary
                  0, 1, 0,
                  0, 0, 1,
                  0,        # pairwise
                  0, 0,
                  0, 0, 0])
    x = (np.array([[1, 1.5, 1.1]]), np.empty((0, 2)))
    assert_equal(crf.inference(x, w), 1)
    # loss augmented inference picks last
    assert_equal(crf.loss_augmented_inference(x, [1], w), 2)

    # with class-weights, loss for class 1 is smaller, loss-augmented inference
    # will find it
    crf = GraphCRF(n_states=3, n_features=3, class_weight=[1, .1, 1])
    assert_equal(crf.loss_augmented_inference(x, [1], w), 1)


def test_directed_graph_chain():
    # check that a directed model actually works differntly in the two
    # directions.  chain of length three, three states 0, 1, 2 which want to be
    # in this order, evidence only in the middle
    x = (np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
         np.array([[0, 1], [1, 2]]))

    w = np.array([1, 0, 0,  # unary
                  0, 1, 0,
                  0, 0, 1,
                  0, 1, 0,  # pairwise
                  0, 0, 1,
                  0, 0, 0])
    crf = GraphCRF(n_states=3, n_features=3, directed=True)
    y = crf.inference(x, w)
    assert_array_equal([0, 1, 2], y)
