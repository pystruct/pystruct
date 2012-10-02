from crf import BinaryGridCRF
from structured_svm import StructuredSVM, SubgradientStructuredSVM
from structured_svm import objective_primal
from toy_datasets import binary

from IPython.core.debugger import Tracer
tracer = Tracer()


def test_primal_dual_binary():
    for C in [1, 100, 100000]:
        for dataset in binary:
            X, Y = dataset(n_samples=1)
            crf = BinaryGridCRF()
            clf = StructuredSVM(problem=crf, max_iter=200, C=C, verbose=0,
                    check_constraints=True)
            clf.fit(X, Y)
            clf2 = SubgradientStructuredSVM(problem=crf, max_iter=200, C=C,
                    verbose=0)
            clf2.fit(X, Y)
            obj = objective_primal(crf, clf.w, X, Y, C)
            # the dual finds the optimum so it might be better
            obj2 = objective_primal(crf, clf2.w, X, Y, C)
            assert(obj <= obj2)
            print("objective difference: %f\n" % (obj2 - obj))
test_primal_dual_binary()
