# this example tells us that we shouldn't let a QP solver do the job of SMO

from time import time
import numpy as np

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC

from pystruct.problems import BinarySVMProblem
from pystruct.learners import StructuredSVM

# do a binary digit classification
digits = load_digits()
X, y = digits.data, digits.target

# make binary task by doing odd vs even numers
y = y % 2
# code as +1 and -1
y = 2 * y - 1
X /= X.max()

X_train, X_test, y_train, y_test = train_test_split(X, y)

pbl = BinarySVMProblem(n_features=X_train.shape[1] + 1)  # add one for bias
svm = StructuredSVM(pbl, verbose=2, check_constraints=False, C=20)

start = time()
# we add a constant 1 feature for the bias
X_train_bias = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
svm.fit(X_train_bias, y_train)
time_svm = time() - start
X_test_bias = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
y_pred = np.hstack(svm.predict(X_test_bias))
print("Score with pystruct toy svm: %f (took %f seconds)"
      % (np.mean(y_pred == y_test), time_svm))

# because of the way I construct psi, we use half the C
libsvm = SVC(kernel='linear', C=10)
start = time()
libsvm.fit(X_train, y_train)
time_libsvm = time() - start
print("Score with sklearn and libsvm: %f (took %f seconds)"
      % (libsvm.score(X_test, y_test), time_libsvm))
