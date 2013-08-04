import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_true
from pystruct.models import GridCRF
from pystruct.learners import NSlackSSVM, SubgradientSSVM
import pystruct.toy_datasets as toy
from pystruct.inference import get_installed

# we always try to get the fastest installed inference method


def test_binary_blocks_cutting_plane():
    #testing cutting plane ssvm on easy binary dataset
    inference_method = get_installed(["qpbo", "ad3", "lp"])[0]
    X, Y = toy.generate_blocks(n_samples=5)
    crf = GridCRF(inference_method=inference_method)
    clf = NSlackSSVM(model=crf, max_iter=20, C=100,
                     check_constraints=True, break_on_bad=False)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_binary_blocks_batches_n_slack():
    #testing cutting plane ssvm on easy binary dataset
    X, Y = toy.generate_blocks(n_samples=5)
    inference_method = get_installed(["qpbo", "ad3", "lp"])[0]
    crf = GridCRF(inference_method=inference_method)
    clf = NSlackSSVM(model=crf, max_iter=20, batch_size=1, C=100)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_binary_blocks_subgradient():
    #testing subgradient ssvm on easy binary dataset
    X, Y = toy.generate_blocks(n_samples=5)
    inference_method = get_installed(["qpbo", "ad3", "lp"])[0]
    crf = GridCRF(inference_method=inference_method)
    clf = SubgradientSSVM(model=crf, C=100, learning_rate=1, decay_exponent=1,
                          momentum=0, decay_t0=10)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_binary_checker_subgradient():
    #testing subgradient ssvm on non-submodular binary dataset
    X, Y = toy.generate_checker(n_samples=10)
    crf = GridCRF()
    clf = SubgradientSSVM(model=crf, max_iter=100, C=100, momentum=.9,
                          learning_rate=0.1)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_binary_ssvm_repellent_potentials():
    # test non-submodular learning with and without positivity constraint
    # dataset is checkerboard
    X, Y = toy.generate_checker()
    inference_method = get_installed(["qpbo", "ad3", "lp"])[0]
    crf = GridCRF(inference_method=inference_method)
    clf = NSlackSSVM(model=crf, max_iter=10, C=100,
                     check_constraints=True)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    # standard crf can predict perfectly
    assert_array_equal(Y, Y_pred)

    submodular_clf = NSlackSSVM(model=crf, max_iter=10, C=100,
                                check_constraints=True,
                                positive_constraint=[4, 5, 6])
    submodular_clf.fit(X, Y)
    Y_pred = submodular_clf.predict(X)
    # submodular crf can not do better than unaries
    for i, x in enumerate(X):
        y_pred_unaries = crf.inference(x, np.array([1, 0, 0, 1, 0, 0, 0]))
        assert_array_equal(y_pred_unaries, Y_pred[i])


def test_binary_ssvm_attractive_potentials():
    # test that submodular SSVM can learn the block dataset
    inference_method = get_installed(["qpbo", "ad3", "lp"])[0]
    X, Y = toy.generate_blocks(n_samples=10)
    crf = GridCRF(inference_method=inference_method)
    submodular_clf = NSlackSSVM(model=crf, max_iter=200, C=100,
                                check_constraints=True,
                                positive_constraint=[5])
    submodular_clf.fit(X, Y)
    Y_pred = submodular_clf.predict(X)
    assert_array_equal(Y, Y_pred)
    assert_true(submodular_clf.w[5] < 0)  # don't ask me about signs
