import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
##from nose.tools import assert_equal, assert_almost_equal, assert_raises
from nose.tools import assert_almost_equal

from pystruct.problems import GraphCRF
from pystruct.inference import inference_lp

w = np.array([1, 1,
              .22,
              0, .22])

# triangle
x_1 = np.array([[0, 1], [1, 0], [.4, .6]])
g_1 = np.array([[0, 1], [1, 2], [2, 0]])
# expected result
y_1 = np.array([1, 0, 1])

# chain
x_2 = np.array([[0, 1], [1, 0], [.4, .6]])
g_2 = np.array([[0, 1], [1, 2]])
# expected result
y_2 = np.array([1, 0, 0])


def test_graph_crf_inference():
    # create two samples with different graphs
    # two states only, pairwise smoothing
    for inference_method in ['qpbo', 'lp', 'ad3', 'dai']:
        crf = GraphCRF(n_states=2, inference_method=inference_method)
        assert_array_equal(crf.inference((x_1, g_1), w), y_1)
        assert_array_equal(crf.inference((x_2, g_2), w), y_2)


def test_graph_crf_continuous_inference():
    for inference_method in ['lp', 'ad3']:
        crf = GraphCRF(n_states=2, inference_method=inference_method)
        assert_array_equal(np.argmax(crf.inference((x_1, g_1), w,
                                                   relaxed=True)[0], axis=-1),
                           y_1)
        assert_array_equal(np.argmax(crf.inference((x_2, g_2), w,
                                                   relaxed=True)[0], axis=-1),
                           y_2)


def test_graph_crf_energy_lp_integral():
    crf = GraphCRF(n_states=2, inference_method='lp')
    unary_params = crf.get_unary_weights(w)
    pairwise_params = crf.get_pairwise_weights(w)
    inf_res, energy_lp = inference_lp(x_1, unary_params, pairwise_params,
                                      edges=g_1, relaxed=True,
                                      return_energy=True, exact=True)
    # integral solution
    assert_array_almost_equal(np.max(inf_res[0], axis=-1), 1)
    y = np.argmax(inf_res[0], axis=-1)
    # energy and psi check out
    assert_almost_equal(energy_lp, -np.dot(w, crf.psi((x_1, g_1), y)))


def test_graph_crf_energy_lp_relaxed():
    crf = GraphCRF(n_states=2, inference_method='lp')
    unary_params = crf.get_unary_weights(w)
    pairwise_params = crf.get_pairwise_weights(w)
    inf_res, energy_lp = inference_lp(x_2, unary_params, pairwise_params,
                                      edges=g_2, relaxed=True,
                                      return_energy=True, exact=True)
    assert_almost_equal(energy_lp, -np.dot(w, crf.psi((x_2, g_2), inf_res)))

    # now with fractional solution
    x = np.array([[0, 0], [0, 0], [0, 0]])
    inf_res, energy_lp = inference_lp(x, unary_params, -pairwise_params,
                                      edges=g_1, relaxed=True,
                                      return_energy=True, exact=True)
    assert_almost_equal(energy_lp, -np.dot(w, crf.psi((x, g_1), inf_res)))


def test_graph_crf_loss_augment():
    x = (x_1, g_1)
    y = y_1
    crf = GraphCRF(n_states=2)
    x_loss_augmented = crf.loss_augment(x, y, w)
    # check that y has same energy with loss and loss-augmented x
    assert_almost_equal(np.dot(w, crf.psi(x, y)),
                        np.dot(w, crf.psi(x_loss_augmented, y)))
    # check that y_2 fulfulls energy + loss condition
    assert_almost_equal(np.dot(w, crf.psi(x, y_2)) + crf.loss(y, y_2),
                        np.dot(w, crf.psi(x_loss_augmented, y_2)))
