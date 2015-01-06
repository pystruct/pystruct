import itertools

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_almost_equal, assert_equal, assert_raises

from pystruct.models import MultiLabelClf
from pystruct.inference import compute_energy


def test_initialization():
    x = np.random.normal(size=(13, 5))
    y = np.random.randint(2, size=(13, 3))
    # no edges make independent model
    model = MultiLabelClf()
    model.initialize(x, y)
    assert_equal(model.n_states, 2)
    assert_equal(model.n_labels, 3)
    assert_equal(model.n_features, 5)
    assert_equal(model.size_joint_feature, 5 * 3)

    # setting and then initializing is no-op
    model = MultiLabelClf(n_features=5, n_labels=3)
    model.initialize(x, y)  # smoketest

    model = MultiLabelClf(n_features=3, n_labels=3)
    assert_raises(ValueError, model.initialize, X=x, Y=y)


def test_multilabel_independent():
    # test inference and energy with independent model
    edges = np.zeros((0, 2), dtype=np.int)
    n_features = 5
    n_labels = 4
    model = MultiLabelClf(n_labels=n_labels, n_features=n_features,
                          edges=edges)
    rnd = np.random.RandomState(0)

    x = rnd.normal(size=5)
    w = rnd.normal(size=n_features * n_labels)
    # test inference
    y = model.inference(x, w)
    y_ = np.dot(w.reshape(n_labels, n_features), x) > 0
    assert_array_equal(y, y_)

    # test joint_feature / energy
    joint_feature = model.joint_feature(x, y)
    energy = compute_energy(model._get_unary_potentials(x, w),
                            model._get_pairwise_potentials(x, w), edges, y)
    assert_almost_equal(energy, np.dot(joint_feature, w))

    # for continuous y
    y_continuous = np.zeros((n_labels, 2))
    y_continuous[np.arange(n_labels), y] = 1
    assert_array_almost_equal(
        joint_feature, model.joint_feature(x, (y_continuous, np.zeros((0, n_labels, n_labels)))))


def test_multilabel_fully():
    # test inference and energy with fully connected model
    n_features = 5
    n_labels = 4
    edges = np.vstack([x for x in itertools.combinations(range(n_labels), 2)])
    model = MultiLabelClf(n_labels=n_labels, n_features=n_features,
                          edges=edges)
    rnd = np.random.RandomState(0)

    x = rnd.normal(size=n_features)
    w = rnd.normal(size=n_features * n_labels + 4 * len(edges))
    y = model.inference(x, w)

    # test joint_feature / energy
    joint_feature = model.joint_feature(x, y)
    energy = compute_energy(model._get_unary_potentials(x, w),
                            model._get_pairwise_potentials(x, w), edges, y)
    assert_almost_equal(energy, np.dot(joint_feature, w))

    # for continuous y
    #y_cont = model.inference(x, w, relaxed=True)
    y_continuous = np.zeros((n_labels, 2))
    pairwise_marginals = []
    for edge in edges:
        # indicator of one of four possible states of the edge
        pw = np.zeros((2, 2))
        pw[y[edge[0]], y[edge[1]]] = 1
        pairwise_marginals.append(pw)

    pairwise_marginals = np.vstack(pairwise_marginals)

    y_continuous[np.arange(n_labels), y] = 1
    assert_array_almost_equal(
        joint_feature, model.joint_feature(x, (y_continuous, pairwise_marginals)))
