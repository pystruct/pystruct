"""
================================
Sequence classifcation benchmark
================================
This is a stripped-down version of the "plot_letters.py" example
targetted to benchmark inference and learning algorithms on chains.
"""
import numpy as np

from pystruct.datasets import load_letters
from pystruct.models import ChainCRF
from pystruct.learners import OneSlackSSVM

abc = "abcdefghijklmnopqrstuvwxyz"

letters = load_letters()
X, y, folds = letters['data'], letters['labels'], letters['folds']
# we convert the lists to object arrays, as that makes slicing much more
# convenient
X, y = np.array(X), np.array(y)
X_train, X_test = X[folds == 1], X[folds != 1]
y_train, y_test = y[folds == 1], y[folds != 1]

# Train linear chain CRF
model = ChainCRF()
ssvm = OneSlackSSVM(model=model, C=.1, tol=0.1, verbose=3, max_iter=20)
ssvm.fit(X_train, y_train)

print("Test score with chain CRF: %f" % ssvm.score(X_test, y_test))
