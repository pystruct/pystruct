import numpy as np
from sklearn.datasets import fetch_mldata

from pystruct.learners import OneSlackSSVM
from pystruct.models import MultiLabelModel

def test_multilabel_yeast_independent():
    # WORK IN PROGRESS!!
    #yeast = fetch_mldata("yeast")
    #X = yeast.data
    #y = yeast.target.toarray().T.astype(np.int)
    ## no edges for the moment
    #edges = np.zeros((0, 2), dtype=np.int)
    #pbl = MultiLabelModel(n_features=X.shape[1], n_labels=y.shape[1],
                            #edges=edges)
    #ssvm = OneSlackSSVM(pbl, verbose=10)
    #ssvm.fit(X, y)
    pass

