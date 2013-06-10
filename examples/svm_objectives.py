"""
====================
SVM objective values
====================
Showing the relation between cutting plane and primal objectives
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split

from pystruct.models import CrammerSingerSVMModel
from pystruct.learners import (NSlackSSVM, OneSlackSSVM,
                               SubgradientSSVM)

# do a binary digit classification
digits = load_digits()
X, y = digits.data, digits.target

X /= X.max()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# we add a constant 1 feature for the bias
X_train_bias = np.hstack([X_train, np.ones((X_train.shape[0], 1))])

pbl = CrammerSingerSVMModel(n_features=X_train_bias.shape[1], n_classes=10)
n_slack_svm = NSlackSSVM(pbl, verbose=0, check_constraints=False, C=20,
                         max_iter=500, batch_size=10)
one_slack_svm = OneSlackSSVM(pbl, verbose=0, check_constraints=False, C=20,
                             max_iter=1000, tol=0.001)
subgradient_svm = SubgradientSSVM(pbl, C=20, learning_rate=0.01, max_iter=300,
                                  decay_exponent=0, momentum=0, verbose=0)

# n-slack cutting plane ssvm
n_slack_svm.fit(X_train_bias, y_train)

## 1-slack cutting plane ssvm
one_slack_svm.fit(X_train_bias, y_train)

# online subgradient ssvm
subgradient_svm.fit(X_train_bias, y_train)

#plt.plot(n_slack_svm.objective_curve_, label="n-slack lower bound")
plt.plot(n_slack_svm.objective_curve_, label="n-slack lower bound")
plt.plot(one_slack_svm.objective_curve_, label="one-slack lower bound")
plt.plot(one_slack_svm.primal_objective_curve_, label="one-slack primal")
plt.plot(subgradient_svm.objective_curve_, label="subgradient")
plt.legend()
plt.show()
