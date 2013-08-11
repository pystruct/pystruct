from numpy.testing import assert_array_equal

from pystruct.models import GridCRF
from pystruct.learners import StructuredPerceptron
from pystruct.datasets import generate_blocks
from pystruct.inference import get_installed


def test_binary_blocks_perceptron_online():
    #testing subgradient ssvm on easy binary dataset
    X, Y = generate_blocks(n_samples=10)
    inference_method = get_installed(['qpbo', 'ad3', 'lp'])[0]
    crf = GridCRF(inference_method=inference_method)
    clf = StructuredPerceptron(model=crf, max_iter=20)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_binary_blocks_perceptron_parallel():
    pass
    #testing subgradient ssvm on easy binary dataset
    #X, Y = generate_blocks(n_samples=10)
    #crf = GridCRF()
    #clf = StructuredPerceptron(model=crf, max_iter=200,
    #batch=True, #n_jobs=-1)
    #clf.fit(X, Y)
    #Y_pred = clf.predict(X)
    #assert_array_equal(Y, Y_pred)
