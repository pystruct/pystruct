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

pbl = BinarySVMProblem(n_features=X_train.shape[1])
svm = StructuredSVM(pbl, verbose=2, check_constraints=False, C=20)

start = time()
svm.fit(X_train, y_train)
time_svm = time() - start
y_pred = np.hstack(svm.predict(X_test))
print("Score with pystruct toy svm: %f (took %f seconds)"
      % (np.mean(y_pred == y_test), time_svm))

# because of the way I construct psi, we use half the C
libsvm = SVC(kernel='linear', C=10)
start = time()
libsvm.fit(X_train, y_train)
time_libsvm = time() - start
print("Score with sklearn and libsvm: %f (took %f seconds)"
      % (libsvm.score(X_test, y_test), time_libsvm))
