import numpy as np
from numpy.testing import assert_array_equal
from sklearn.utils.testing import assert_equal, assert_true

from pystruct.models import LatentGridCRF, LatentDirectionalGridCRF
from pystruct.learners import (LatentSSVM, NSlackSSVM, OneSlackSSVM,
                               SubgradientSSVM)

import pystruct.toy_datasets as toy


def test_with_crosses_perfect_init():
    # very simple dataset. k-means init is perfect
    for n_states_per_label in [2, [1, 2]]:
        # test with 2 states for both foreground and background,
        # as well as with single background state
        #for inference_method in ['ad3', 'qpbo', 'lp']:
        for inference_method in ['ad3']:
            X, Y = toy.generate_crosses(n_samples=10, noise=5, n_crosses=1,
                                        total_size=8)
            n_labels = 2
            crf = LatentGridCRF(n_labels=n_labels,
                                n_states_per_label=n_states_per_label,
                                inference_method=inference_method)
            clf = LatentSSVM(OneSlackSSVM(model=crf, max_iter=50, C=10. ** 5,
                                          verbose=2, check_constraints=True,
                                          n_jobs=-1, break_on_bad=True))
            clf.fit(X, Y)
            Y_pred = clf.predict(X)
            assert_array_equal(np.array(Y_pred), Y)
            assert_equal(clf.score(X, Y), 1)


def test_with_crosses_base_svms():
    # very simple dataset. k-means init is perfect
    n_labels = 2
    crf = LatentGridCRF(n_labels=n_labels, n_states_per_label=[1, 2],
                        inference_method='lp')
    one_slack = OneSlackSSVM(crf)
    n_slack = NSlackSSVM(crf)
    subgradient = SubgradientSSVM(crf, max_iter=150, learning_rate=5)

    for base_ssvm in [one_slack, n_slack, subgradient]:
        base_ssvm.C = 10. ** 5
        base_ssvm.n_jobs = -1
        X, Y = toy.generate_crosses(n_samples=10, noise=5, n_crosses=1,
                                    total_size=8)
        clf = LatentSSVM(base_ssvm=base_ssvm)
        clf.fit(X, Y)
        Y_pred = clf.predict(X)
        assert_array_equal(np.array(Y_pred), Y)
        assert_equal(clf.score(X, Y), 1)


def test_with_crosses_bad_init():
    # use less perfect initialization
    X, Y = toy.generate_crosses(n_samples=20, noise=5, n_crosses=1,
                                total_size=8)
    X_test, Y_test = X[10:], Y[10:]
    X, Y = X[:10], Y[:10]
    n_labels = 2
    crf = LatentGridCRF(n_labels=n_labels, n_states_per_label=2,
                        inference_method='lp')
    H_init = crf.init_latent(X, Y)

    mask = np.random.uniform(size=H_init.shape) > .7
    H_init[mask] = 2 * (H_init[mask] / 2)

    one_slack = OneSlackSSVM(crf, inactive_threshold=1e-8, cache_tol=.0001,
                             inference_cache=50, max_iter=10000)
    n_slack = NSlackSSVM(crf)
    subgradient = SubgradientSSVM(crf, max_iter=150, learning_rate=5)

    for base_ssvm in [one_slack, n_slack, subgradient]:
        base_ssvm.C = 10. ** 3
        base_ssvm.n_jobs = -1
        clf = LatentSSVM(base_ssvm)

        clf.fit(X, Y, H_init=H_init)
        Y_pred = clf.predict(X)

        assert_array_equal(np.array(Y_pred), Y)
        # test that score is not always 1
        assert_true(.98 < clf.score(X_test, Y_test) < 1)


def test_directional_bars():
    for inference_method in ['ad3']:
        X, Y = toy.generate_easy(n_samples=10, noise=5, box_size=2,
                                 total_size=6, seed=1)
        n_labels = 2
        crf = LatentDirectionalGridCRF(n_labels=n_labels,
                                       n_states_per_label=[1, 4],
                                       inference_method=inference_method)
        clf = LatentSSVM(OneSlackSSVM(model=crf, max_iter=500, C=10.,
                                      verbose=3, n_jobs=-1,
                                      inference_cache=50))
        clf.fit(X, Y)
        Y_pred = clf.predict(X)

        assert_array_equal(np.array(Y_pred), Y)


def test_switch_to_ad3():
    # smoketest only
    # test if switching between qpbo and ad3 works inside latent svm
    # use less perfect initialization
    X, Y = toy.generate_crosses(n_samples=20, noise=5, n_crosses=1,
                                total_size=8)
    X_test, Y_test = X[10:], Y[10:]
    X, Y = X[:10], Y[:10]
    n_labels = 2
    crf = LatentGridCRF(n_labels=n_labels, n_states_per_label=2,
                        inference_method='qpbo')
    H_init = crf.init_latent(X, Y)

    mask = np.random.uniform(size=H_init.shape) > .7
    H_init[mask] = 2 * (H_init[mask] / 2)

    base_ssvm = OneSlackSSVM(crf, inactive_threshold=1e-8, cache_tol=.0001,
                             inference_cache=50, max_iter=10000,
                             switch_to="ad3")
    base_ssvm.C = 10. ** 3
    base_ssvm.n_jobs = -1
    clf = LatentSSVM(base_ssvm)

    clf.fit(X, Y, H_init=H_init)
    Y_pred = clf.predict(X)

    assert_array_equal(np.array(Y_pred), Y)
    # test that score is not always 1
    assert_true(.98 < clf.score(X_test, Y_test) < 1)
    assert_equal(clf.model.inference_method, "qpbo")
