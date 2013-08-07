from sklearn.datasets import load_svmlight_file

import numpy as np
import itertools

#from multi_label import chow_liu_tree

X_train, y_train = load_svmlight_file("scene/scene_train", multilabel=True)
X_test, y_test = load_svmlight_file("scene/scene_test", multilabel=True)


X_train.shape
X_test.shape

from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import hamming_loss

clf = OneVsRestClassifier(LinearSVC())

from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

y_train.mean(axis=0)
y_test.mean(axis=0)

X_train = X_train.toarray()
X_test = X_test.toarray()

#clf.fit(X_train, y_train)
#hamming_loss(y_train, clf.predict(X_train))
#hamming_loss(y_test, clf.predict(X_test))


param_grid = {'estimator__C': 10. ** np.arange(-4, 2)}
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)
grid = GridSearchCV(clf, param_grid=param_grid, verbose=3,
                    loss_func=hamming_loss, cv=5)

#grid.fit(X_train, y_train)
#hamming_loss(y_train, grid.predict(X_train))

#clf.set_params(estimator__C=0.01)
#clf.fit(X_train, y_train)

#hamming_loss(y_train, clf.predict(X_train))
#hamming_loss(y_test, clf.predict(X_test))

from pystruct.learners import OneSlackSSVM
from pystruct.models import MultiLabelModel

# <codecell>
edges = np.vstack([x for x in itertools.combinations(range(6), 2)])

ssvm = OneSlackSSVM(MultiLabelModel(edges=edges,
                                    inference_method=("ad3",
                                                      {'branch_and_bound':
                                                       True})),
                    verbose=2, C=0.1, inference_cache=50)


ssvm.fit(X_train, y_train)

print(hamming_loss(y_test, np.vstack(ssvm.predict(X_test))))
print(hamming_loss(y_train, np.vstack(ssvm.predict(X_train))))

from IPython.core.debugger import Tracer
Tracer()()

#final primal objective: 19.288704 gap: 0.011139
# C=0.01
#0.115105908584
#0.100467932838

# C=0.1
#0.111900780379
#0.0659234792183

# C=0.1
# qpbo full edges
# 0.105490523969
# 0.0879438480595

# ad3
# 0.10604793757
# 0.088769611891
