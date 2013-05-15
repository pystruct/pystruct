import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
##from sklearn.utils.testing import assert_equal, assert_almost_equal, assert_raises
from sklearn.utils.testing import assert_almost_equal, assert_equal

from pystruct.models import GraphCRF, EdgeTypeGraphCRF, EdgeFeatureGraphCRF


w = np.array([1, 0,  # unary
              0, 1,
              .22,  # pairwise
              0, .22])

# for directional CRF with non-symmetric weights
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


def test_graph_crf_inference():
    # create two samples with different graphs
    # two states only, pairwise smoothing
    for inference_method in ['qpbo', 'lp', 'ad3', 'dai', 'ogm']:
        crf = GraphCRF(n_states=2, inference_method=inference_method)
        assert_array_equal(crf.inference((x_1, g_1), w), y_1)
        assert_array_equal(crf.inference((x_2, g_2), w), y_2)

    print crf.get_pairwise_potentials((x_1, g_1),
                                      w)



def test_edge_feature_graph_crf() :
    g_ef = [g_1, np.array([[0,1],[1,0],[1,1]])]
    X = (x_1, g_ef)

    w = np.array([1, 0,  # unary
                  0, 1,
                  .22, .88])  # pairwise


    
    print X
    crf = EdgeFeatureGraphCRF(n_states=2, inference_method='qpbo',
                              n_edge_features=2)

    print crf.get_pairwise_potentials(X, w)
        

def test_edge_type_graph_crf():
    # create two samples with different graphs
    # two states only, pairwise smoothing

    # all edges are of the first type. should do the same as GraphCRF
    # if we make w symmetric
    for inference_method in ['qpbo', 'lp', 'ad3', 'dai', 'ogm']:
        crf = EdgeTypeGraphCRF(n_states=2, inference_method=inference_method,
                               n_edge_types=1)
        assert_array_equal(crf.inference((x_1, [g_1]), w_sym), y_1)
        assert_array_equal(crf.inference((x_2, [g_2]), w_sym), y_2)



    # same, only with two edge types and no edges of second type
    w_sym_ = np.array([1, 0,    # unary
                      0, 1,
                      .22, 0,  # pairwise
                      0, .22,
                      2, -1,   # second edge type, doesn't exist
                      -1, 3])
    for inference_method in ['qpbo', 'lp', 'ad3', 'dai', 'ogm']:
        crf = EdgeTypeGraphCRF(n_states=2, inference_method=inference_method,
                               n_edge_types=2)
        assert_array_equal(crf.inference((x_1,
                                          [g_1, np.zeros((0, 2),
                                                         dtype=np.int)]),
                                         w_sym_), y_1)
        assert_array_equal(crf.inference((x_2, [g_2, np.zeros((0, 2),
                                                              dtype=np.int)]),
                                         w_sym_), y_2)


    print crf.get_pairwise_potentials((x_2, [g_2, np.zeros((0, 2),
                                                           dtype=np.int)]),
                                      w_sym_)

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
    inf_res, energy_lp = crf.inference((x_1, g_1), w, relaxed=True,
                                       return_energy=True)
    # integral solution
    assert_array_almost_equal(np.max(inf_res[0], axis=-1), 1)
    y = np.argmax(inf_res[0], axis=-1)
    # energy and psi check out
    assert_almost_equal(energy_lp, -np.dot(w, crf.psi((x_1, g_1), y)))


def test_graph_crf_energy_lp_relaxed():
    crf = GraphCRF(n_states=2, inference_method='lp')
    for i in xrange(10):
        w_ = np.random.uniform(size=w.shape)
        inf_res, energy_lp = crf.inference((x_1, g_1), w_, relaxed=True,
                                           return_energy=True)
        assert_almost_equal(energy_lp,
                            -np.dot(w_, crf.psi((x_1, g_1), inf_res)))

    # now with fractional solution
    x = np.array([[0, 0], [0, 0], [0, 0]])
    inf_res, energy_lp = crf.inference((x, g_1), w, relaxed=True,
                                       return_energy=True)
    assert_almost_equal(energy_lp, -np.dot(w, crf.psi((x, g_1), inf_res)))


def test_graph_crf_loss_augment():
    x = (x_1, g_1)
    y = y_1
    crf = GraphCRF(n_states=2, inference_method='lp')
    y_hat, energy = crf.loss_augmented_inference(x, y, w, return_energy=True)
    # check that y_hat fulfills energy + loss condition
    assert_almost_equal(np.dot(w, crf.psi(x, y_hat)) + crf.loss(y, y_hat),
                        -energy)


def test_edge_type_graph_crf_energy_lp_integral():
    # same test as for graph crf above, using single edge type
    crf = EdgeTypeGraphCRF(n_states=2, inference_method='lp', n_edge_types=1)
    inf_res, energy_lp = crf.inference((x_1, [g_1]), w_sym, relaxed=True,
                                       return_energy=True)
    # integral solution
    assert_array_almost_equal(np.max(inf_res[0], axis=-1), 1)
    y = np.argmax(inf_res[0], axis=-1)
    # energy and psi check out
    assert_almost_equal(energy_lp, -np.dot(w_sym, crf.psi((x_1, [g_1]), y)))


def test_edge_type_graph_crf_energy_lp_relaxed():
    # same test as for graph crf above, using single edge type
    crf = EdgeTypeGraphCRF(n_states=2, inference_method='lp',
                           n_edge_types=1)
    for i in xrange(10):
        w_ = np.random.uniform(size=w_sym.shape)
        inf_res, energy_lp = crf.inference((x_1, [g_1]), w_, relaxed=True,
                                           return_energy=True)
        assert_almost_equal(energy_lp,
                            -np.dot(w_, crf.psi((x_1, [g_1]), inf_res)))

    # now with fractional solution
    x = np.array([[0, 0], [0, 0], [0, 0]])
    inf_res, energy_lp = crf.inference((x, [g_1]), w_sym, relaxed=True,
                                       return_energy=True)
    assert_almost_equal(energy_lp,
                        -np.dot(w_sym, crf.psi((x, [g_1]), inf_res)))


def test_graph_crf_class_weights():
    # no edges
    crf = GraphCRF(n_states=3, n_features=3, inference_method='dai')
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
    crf = GraphCRF(n_states=3, n_features=3, inference_method='dai',
                   class_weight=[1, .1, 1])
    assert_equal(crf.loss_augmented_inference(x, [1], w), 1)
