
#import numpy as np
from numpy.testing import assert_array_equal
#from nose.tools import assert_true
from pystruct.crf import GraphCRF
from pystruct.structured_svm import StructuredSVM
import pystruct.toy_datasets as toy
from pystruct.inference_methods import _make_grid_edges


def test_binary_blocks_cutting_plane():
    #testing cutting plane ssvm on easy binary dataset
    # generate graphs explicitly for each example
    for inference_method in ["dai", "lp", "qpbo", "ad3"]:
        X, Y = toy.generate_blocks(n_samples=3)
        crf = GraphCRF(inference_method=inference_method)
        clf = StructuredSVM(problem=crf, max_iter=20, C=100, verbose=0,
                            check_constraints=True, break_on_bad=False,
                            n_jobs=1)
        x1, x2, x3 = X
        y1, y2, y3 = Y
        # delete some rows to make it more fun
        x1, y1 = x1[:, -1], y1[:, -1]
        x2, y2 = x2[:-1], y2[:-1]

        g1 = _make_grid_edges(x1)
        g2 = _make_grid_edges(x2)
        g3 = _make_grid_edges(x2)
        X = [(x1, g1), (x2, g2), (x3, g3)]
        Y = [y1, y2, y3]

        clf.fit(X, Y)
        Y_pred = clf.predict(X)
        assert_array_equal(Y, Y_pred)
