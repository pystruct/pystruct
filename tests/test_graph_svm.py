import numpy as np
from numpy.testing import assert_array_equal
#from nose.tools import assert_true
from pystruct.problems import GraphCRF
from pystruct.learners import StructuredSVM
import pystruct.toy_datasets as toy
from pystruct.utils import make_grid_edges


def test_binary_blocks_cutting_plane():
    #testing cutting plane ssvm on easy binary dataset
    # generate graphs explicitly for each example
    for inference_method in ["dai", "lp", "qpbo", "ad3"]:
        print("testing %s" % inference_method)
        X, Y = toy.generate_blocks(n_samples=3)
        crf = GraphCRF(inference_method=inference_method)
        clf = StructuredSVM(problem=crf, max_iter=20, C=100, verbose=0,
                            check_constraints=True, break_on_bad=False,
                            n_jobs=1)
        #from IPython.core.debugger import Tracer
        #Tracer()()
        x1, x2, x3 = X
        y1, y2, y3 = Y
        n_states = len(np.unique(Y))
        # delete some rows to make it more fun
        x1, y1 = x1[:, :-1], y1[:, :-1]
        x2, y2 = x2[:-1], y2[:-1]
        # generate graphs
        X_ = [x1, x2, x3]
        G = [make_grid_edges(x) for x in X_]

        # reshape / flatten x and y
        X_ = [x.reshape(-1, n_states) for x in X_]
        Y = [y.ravel() for y in [y1, y2, y3]]

        X = zip(X_, G)

        clf.fit(X, Y)
        Y_pred = clf.predict(X)
        for y, y_pred in zip(Y, Y_pred):
            assert_array_equal(y, y_pred)
