import itertools

import numpy as np
from scipy import sparse

from sklearn.metrics import hamming_loss
from sklearn.datasets import fetch_mldata
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mutual_info_score
from sklearn.utils.mst import minimum_spanning_tree

from pystruct.learners import OneSlackSSVM
from pystruct.models import MultiLabelModel
#from pystruct.utils import SaveLogger


def chow_liu_tree(y):
    # compute mutual information using sklearn
    mi = np.zeros((14, 14))
    for i in xrange(14):
        for j in xrange(14):
            mi[i, j] = mutual_info_score(y[:, i], y[:, j])
    mst = minimum_spanning_tree(sparse.csr_matrix(-mi))
    return mst


def my_hamming(y_train, y_pred):
    return hamming_loss(y_train, np.vstack(y_pred))

yeast = fetch_mldata("yeast")

# for both, mine and ovr, C=.1 seems good!

X = yeast.data
X = np.hstack([X, np.ones((X.shape[0], 1))])
y = yeast.target.toarray().astype(np.int).T


X_train, X_test = X[:1500], X[1500:]
y_train, y_test = y[:1500], y[1500:]


X_train.shape

#import itertools
edges = np.vstack([x for x in itertools.combinations(range(14), 2)])
#edges = np.zeros((0, 2), dtype=np.int)

model = MultiLabelModel(14, X.shape[1], edges=edges, inference_method='qpbo')

#logger = SaveLogger('multi_label_fully_switch_to_dai.pickle', save_every=20)
ssvm = OneSlackSSVM(model, inference_cache=50, verbose=1, n_jobs=-1, C=.01,
                    show_loss_every=20, max_iter=10000, tol=0.01,
                    switch_to='ad3bb')

#param_grid = {'C': 10. ** np.arange(-3, 1)}

#grid = GridSearchCV(ssvm, loss_func=my_hamming, cv=5, n_jobs=1, verbose=10,
                    #param_grid=param_grid)
#grid.fit(X_train, y_train)
#from IPython.core.debugger import Tracer
#Tracer()()
ssvm.fit(X_train, y_train)
print(ssvm.score(X_train, y_train))
print(ssvm.score(X_test, y_test))
print(my_hamming(y_test, ssvm.predict(X_test)))

from IPython.core.debugger import Tracer
Tracer()()
