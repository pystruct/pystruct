
from numpy.testing import assert_array_equal

from pystruct.models import GridCRF

from pystruct.datasets import generate_blocks_multinomial

from pystruct.learners import FrankWolfeSSVM


def test_multinomial_blocks_frankwolfe():
    X, Y = generate_blocks_multinomial(n_samples=50, noise=0.4,
                                       seed=0)
    crf = GridCRF(inference_method='qpbo')
    clf = FrankWolfeSSVM(model=crf, C=1, line_search=True,
                         batch_mode=False, dual_check_every=500)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)
