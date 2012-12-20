import numpy as np
from numpy.testing import assert_array_equal

from pystruct.latent_crf import LatentGridCRF, LatentDirectionalGridCRF
from pystruct.latent_structured_svm import LatentSSVM

import pystruct.toy_datasets as toy


def test_with_crosses():
    # very simple dataset. k-means init is perfect
    X, Y = toy.generate_crosses(n_samples=10, noise=5, n_crosses=1,
                                total_size=8)
    n_labels = 2
    crf = LatentGridCRF(n_labels=n_labels, n_states_per_label=2,
                        inference_method='lp')
    clf = LatentSSVM(problem=crf, max_iter=50, C=10. ** 5, verbose=2,
                     check_constraints=True, n_jobs=-1, break_on_bad=True,
                     plot=False)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(np.array(Y_pred) / 2, Y)


def test_with_crosses_bad_init():
    # use less perfect initialization
    X, Y = toy.generate_crosses(n_samples=10, noise=5, n_crosses=1,
                                total_size=8)
    n_labels = 2
    crf = LatentGridCRF(n_labels=n_labels, n_states_per_label=2,
                        inference_method='lp')
    clf = LatentSSVM(problem=crf, max_iter=50, C=10. ** 5, verbose=2,
                     check_constraints=True, n_jobs=-1, break_on_bad=True,
                     plot=False)
    H_init = crf.init_latent(X, Y)

    mask = np.random.uniform(size=H_init.shape) > .7
    H_init[mask] = 2 * (H_init[mask] / 2)
    clf.fit(X, Y, H_init=H_init)
    Y_pred = clf.predict(X)

    assert_array_equal(np.array(Y_pred) / 2, Y)


def test_directional_bars():
    # use less perfect initialization
    X, Y = toy.generate_easy(n_samples=10, noise=5, box_size=2, total_size=6)
    n_labels = 2
    crf = LatentDirectionalGridCRF(n_labels=n_labels, n_states_per_label=4,
                                   inference_method='lp')
    clf = LatentSSVM(problem=crf, max_iter=50, C=10. ** 5, verbose=2,
                     check_constraints=True, n_jobs=-1, break_on_bad=True,
                     plot=False)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)

    assert_array_equal(np.array(Y_pred) / 4, Y)
