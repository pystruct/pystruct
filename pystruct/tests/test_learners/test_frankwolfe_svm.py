
import numpy as np
from numpy.testing import assert_array_equal

from pystruct.models import GridCRF

from pystruct.datasets import generate_blocks_multinomial

from pystruct.learners import FrankWolfeSSVM, OneSlackSSVM


def test_multinomial_blocks_frankwolfe():
    X, Y = generate_blocks_multinomial(n_samples=50, noise=0.5,
                                       seed=0)
    n_labels = len(np.unique(Y))
    crf = GridCRF(n_states=n_labels, inference_method=('ad3', {'branch_and_bound': True}))
    clf = FrankWolfeSSVM(model=crf, max_iter=5, C=1, line_search=False,
                         batch_mode=False, dual_check_every=1)
    #clf = OneSlackSSVM(model=crf, verbose=3, C=1, inference_cache=50)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)
