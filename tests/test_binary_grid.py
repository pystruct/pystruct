import numpy as np
from numpy.testing import assert_array_equal
from sklearn.utils.testing import assert_true
from pystruct.models import GridCRF
from pystruct.learners import StructuredSVM, SubgradientSSVM
import pystruct.toy_datasets as toy


def test_binary_blocks_cutting_plane():
    #testing cutting plane ssvm on easy binary dataset
    for inference_method in ["dai", "lp", "qpbo", "ad3"]:
        X, Y = toy.generate_blocks(n_samples=5)
        crf = GridCRF(inference_method=inference_method)
        clf = StructuredSVM(model=crf, max_iter=20, C=100, verbose=0,
                            check_constraints=True, break_on_bad=False,
                            n_jobs=-1)
        clf.fit(X, Y)
        Y_pred = clf.predict(X)
        assert_array_equal(Y, Y_pred)


def test_binary_blocks_batches_n_slack():
    #testing cutting plane ssvm on easy binary dataset
    X, Y = toy.generate_blocks(n_samples=5)
    crf = GridCRF(inference_method='lp')
    clf = StructuredSVM(model=crf, max_iter=20, C=100, verbose=0,
                        check_constraints=True, break_on_bad=False,
                        n_jobs=1, batch_size=1)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_binary_blocks_subgradient():
    #testing subgradient ssvm on easy binary dataset
    X, Y = toy.generate_blocks(n_samples=10)
    crf = GridCRF()
    clf = SubgradientSSVM(model=crf, max_iter=200, C=100, verbose=10,
                          learning_rate=0.1, n_jobs=-1)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_binary_checker_subgradient():
    #testing subgradient ssvm on non-submodular binary dataset
    X, Y = toy.generate_checker(n_samples=10)
    crf = GridCRF()
    clf = SubgradientSSVM(model=crf, max_iter=100, C=100, verbose=0,
                          momentum=.9, learning_rate=0.1, n_jobs=-1)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_binary_ssvm_repellent_potentials():
    # test non-submodular learning with and without positivity constraint
    # dataset is checkerboard
    X, Y = toy.generate_checker()
    for inference_method in ["lp", "qpbo", "ad3"]:
        crf = GridCRF(inference_method=inference_method)
        clf = StructuredSVM(model=crf, max_iter=10, C=100, verbose=0,
                            check_constraints=True, n_jobs=-1)
        clf.fit(X, Y)
        Y_pred = clf.predict(X)
        # standard crf can predict perfectly
        assert_array_equal(Y, Y_pred)

        submodular_clf = StructuredSVM(model=crf, max_iter=10, C=100,
                                       verbose=0, check_constraints=True,
                                       positive_constraint=[4, 5, 6],
                                       n_jobs=-1)
        submodular_clf.fit(X, Y)
        Y_pred = submodular_clf.predict(X)
        # submodular crf can not do better than unaries
        for i, x in enumerate(X):
            y_pred_unaries = crf.inference(x, np.array([1, 0, 0, 1, 0, 0, 0]))
            assert_array_equal(y_pred_unaries, Y_pred[i])


def test_binary_ssvm_attractive_potentials():
    # test that submodular SSVM can learn the block dataset
    X, Y = toy.generate_blocks(n_samples=10)
    crf = GridCRF()
    submodular_clf = StructuredSVM(model=crf, max_iter=200, C=100,
                                   verbose=1, check_constraints=True,
                                   positive_constraint=[5], n_jobs=-1)
    submodular_clf.fit(X, Y)
    Y_pred = submodular_clf.predict(X)
    assert_array_equal(Y, Y_pred)
    assert_true(submodular_clf.w[5] < 0)  # don't ask me about signs
