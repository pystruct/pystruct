from numpy.testing import assert_array_equal
from crf import BinaryGridCRF
from structured_svm import StructuredSVM, SubgradientStructuredSVM
from structured_svm import objective_primal
from toy_datasets import binary

from IPython.core.debugger import Tracer
tracer = Tracer()


def test_primal_dual_binary():
    for dataset in binary:
        X, Y = dataset(n_samples=1)
        crf = BinaryGridCRF()
        C = 1000000
        clf = StructuredSVM(problem=crf, max_iter=200, C=C, verbose=0,
                check_constraints=True)
        clf.fit(X, Y)
        clf2 = SubgradientStructuredSVM(problem=crf, max_iter=200, C=C,
                verbose=0)
        clf2.fit(X, Y)
        obj = objective_primal(crf, clf.w, X, Y, C)
        obj2 = objective_primal(crf, clf2.w, X, Y, C)
        assert(obj <= obj2)
        print("objective difference: %f" % (obj2 - obj))
        print(clf.predict(X))
        print(clf2.predict(X))
        print("=" * 20)
        #assert_array_equal(clf.predict(X), clf2.predict(X))

test_primal_dual_binary()
