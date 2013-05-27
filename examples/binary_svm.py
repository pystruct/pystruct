"""
==================
Binary SVM as SSVM
==================
Example of training binary SVM using n-slack QP, 1-slack QP, SGD and
SMO (libsvm).  Our 1-slack QP does surprisingly well!
"""

from time import time
import numpy as np

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC

from pystruct.models import BinarySVMModel
from pystruct.learners import (StructuredSVM, OneSlackSSVM,
                               SubgradientSSVM)

# do a binary digit classification
digits = load_digits()
X, y = digits.data, digits.target

# make binary task by doing odd vs even numers
y = y % 2
# code as +1 and -1
y = 2 * y - 1
X /= X.max()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pbl = BinarySVMModel(n_features=X_train.shape[1] + 1)  # add one for bias
n_slack_svm = StructuredSVM(pbl, verbose=0, check_constraints=False, C=10,
                            batch_size=-1)
one_slack_svm = OneSlackSSVM(pbl, verbose=10, check_constraints=False, C=10,
                             max_iter=1000, tol=0.1)
subgradient_svm = SubgradientSSVM(pbl, C=10, learning_rate=0.1, max_iter=100,
                                  decay_exponent=0, batch_size=10, verbose=10)

# we add a constant 1 feature for the bias
X_train_bias = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_test_bias = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

# n-slack cutting plane ssvm
start = time()
n_slack_svm.fit(X_train_bias, y_train)
time_n_slack_svm = time() - start
y_pred = np.hstack(n_slack_svm.predict(X_test_bias))
print("Score with pystruct n-slack ssvm: %f (took %f seconds)"
      % (np.mean(y_pred == y_test), time_n_slack_svm))

## 1-slack cutting plane ssvm
start = time()
one_slack_svm.fit(X_train_bias, y_train)
time_one_slack_svm = time() - start
y_pred = np.hstack(one_slack_svm.predict(X_test_bias))
print("Score with pystruct 1-slack ssvm: %f (took %f seconds)"
      % (np.mean(y_pred == y_test), time_one_slack_svm))

# online subgradient ssvm
start = time()
subgradient_svm.fit(X_train_bias, y_train)
time_subgradient_svm = time() - start
y_pred = np.hstack(subgradient_svm.predict(X_test_bias))

print("Score with pystruct subgradient ssvm: %f (took %f seconds)"
      % (np.mean(y_pred == y_test), time_subgradient_svm))

libsvm = SVC(kernel='linear', C=10)
start = time()
libsvm.fit(X_train, y_train)
time_libsvm = time() - start
print("Score with sklearn and libsvm: %f (took %f seconds)"
      % (libsvm.score(X_test, y_test), time_libsvm))
