from numpy.testing import assert_array_equal
from nose.tools import assert_greater
from pystruct.models import GridCRF
from pystruct.learners import StructuredPerceptron
import pystruct.toy_datasets as toy


def test_binary_blocks():
    X, Y = toy.generate_blocks(n_samples=10)
    crf = GridCRF()
    clf = StructuredPerceptron(model=crf, max_iter=40)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_multinomial_blocks():
    X, Y = toy.generate_blocks_multinomial(n_samples=10, noise=0.3, seed=0)
    crf = GridCRF(n_states=X.shape[-1])
    clf = StructuredPerceptron(model=crf, max_iter=10)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    assert_array_equal(Y, Y_pred)


def test_averaged():
    # Under a lot of noise, averaging helps.  This fails with less noise.
    X, Y = toy.generate_blocks_multinomial(n_samples=20, noise=2, seed=0)
    X_train, Y_train = X[:10], Y[:10]
    X_test, Y_test = X[10:], Y[10:]
    crf = GridCRF(n_states=X.shape[-1])
    clf = StructuredPerceptron(model=crf, max_iter=10)
    clf.fit(X_train, Y_train)
    no_avg_test = clf.score(X_test, Y_test)
    clf.set_params(average=True)
    clf.fit(X_train, Y_train)
    avg_test = clf.score(X_test, Y_test)
    assert_greater(avg_test, no_avg_test)
