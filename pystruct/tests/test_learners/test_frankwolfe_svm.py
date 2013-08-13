from tempfile import mkstemp

import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_less

from sklearn.datasets import load_iris

from pystruct.models import GridCRF, GraphCRF
from pystruct.learners import SubgradientSSVM

from pystruct.datasets import (generate_blocks_multinomial,
                               generate_checker_multinomial, generate_blocks)
from pystruct.utils import SaveLogger, train_test_split, find_constraint

from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal, assert_equal)
from sklearn.datasets import make_blobs
from pystruct.models import MultiClassClf

from frankwolfe_ssvm import FrankWolfeSSVM


def test_multinomial_blocks_frankwolfe():
    X, Y = generate_blocks_multinomial(n_samples=50, noise=0.1,
                                       seed=1)
    n_labels = len(np.unique(Y))
    crf = GridCRF(n_states=n_labels)
    clf = FrankWolfeSSVM(model=crf, max_iter=4, C=10, line_search=True, batch_mode=False, dual_check_every=200)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)
    

    
if __name__ == "__main__":
    test_multinomial_blocks_frankwolfe()


