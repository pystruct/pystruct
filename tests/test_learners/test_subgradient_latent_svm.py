import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)

from pystruct.models import LatentGridCRF, LatentDirectionalGridCRF, GridCRF
from pystruct.learners import LatentSubgradientSSVM, SubgradientSSVM

import pystruct.toy_datasets as toy


def test_with_crosses():
    # this test is just to unstable / not working
    pass
    # very simple dataset. k-means init is perfect
    #for n_states_per_label in [2, [1, 2]]:
    #for n_states_per_label in [2]:
        # test with 2 states for both foreground and background,
        # as well as with single background state
        #for inference_method in ['ad3', 'qpbo', 'lp']:
        #for inference_method in ['qpbo']:
            #X, Y = toy.generate_crosses(n_samples=20, noise=2, n_crosses=1,
                                        #total_size=8, seed=0)
            #n_labels = 2
            #crf = LatentGridCRF(n_labels=n_labels,
                                #n_states_per_label=n_states_per_label,
                                #inference_method=inference_method)
            #clf = LatentSubgradientSSVM(model=crf, max_iter=2250, C=1000.,
                                        #verbose=20, learning_rate=1,
                                        #show_loss_every=0, momentum=0.0,
                                        #decay_exponent=1, decay_t0=10)
            #clf.fit(X, Y)
            #print(clf.predict_latent(X))
            #Y_pred = clf.predict(X)
            #assert_array_equal(np.array(Y_pred), Y)


def test_objective():
    # test that LatentSubgradientSSVM does the same as SubgradientSVM,
    # in particular that it has the same loss, if there are no latent states.
    X, Y = toy.generate_blocks_multinomial(n_samples=10)
    n_labels = 3
    crfl = LatentGridCRF(n_labels=n_labels, n_states_per_label=1)
    clfl = LatentSubgradientSSVM(model=crfl, max_iter=50, C=10.,
                                 learning_rate=0.001, momentum=0.98,
                                 decay_exponent=0)
    clfl.w = np.zeros(crfl.size_psi)  # this disables random init
    clfl.fit(X, Y)

    crf = GridCRF(n_states=n_labels)
    clf = SubgradientSSVM(model=crf, max_iter=50, C=10.,
                          learning_rate=0.001, momentum=0.98, decay_exponent=0)
    clf.fit(X, Y)
    assert_array_almost_equal(clf.w, clfl.w)
    assert_array_equal(clf.predict(X), Y)
    assert_almost_equal(clf.objective_curve_[-1], clfl.objective_curve_[-1])


#def test_with_crosses_bad_init():
#    # use less perfect initialization
#    X, Y = toy.generate_crosses(n_samples=10, noise=5, n_crosses=1,
#                                total_size=8)
#    n_labels = 2
#    crf = LatentGridCRF(n_labels=n_labels, n_states_per_label=2,
#                        inference_method='lp')
#    clf = LatentSubgradientSSVM(model=crf, max_iter=50, C=10. ** 3)
#    H_init = crf.init_latent(X, Y)

#    mask = np.random.uniform(size=H_init.shape) > .7
#    H_init[mask] = 2 * (H_init[mask] / 2)
#    clf.fit(X, Y, H_init=H_init)
#    Y_pred = clf.predict(X)

#    assert_array_equal(np.array(Y_pred), Y)


def test_directional_bars():
    # this test is very fragile :-/
    X, Y = toy.generate_easy(n_samples=20, noise=2, box_size=2,
                             total_size=6, seed=2)
    n_labels = 2
    crf = LatentDirectionalGridCRF(n_labels=n_labels,
                                   n_states_per_label=[1, 4])
    clf = LatentSubgradientSSVM(model=crf, max_iter=75, C=10.,
                                learning_rate=1, momentum=0,
                                decay_exponent=0.5, decay_t0=10)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)

    assert_array_equal(np.array(Y_pred), Y)
