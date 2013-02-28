# a Latent CRF with one node is the same as a latent multiclass SVM
# Using the latent variables, we can learn non-linear problems

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits

from pystruct.problems import GraphCRF, LatentGraphCRF
from pystruct.learners import StructuredSVM, LatentSSVM

# do a binary digit classification
digits = load_digits()
X, y_org = digits.data, digits.target

# make binary task by doing odd vs even numers
y = y_org % 2
X /= X.max()

# make each example into a tuple of a single feature vector and an empty edge
# list
X_ = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]
Y = y.reshape(-1, 1)

X_train_, X_test_, X_train, X_test, y_train, y_test, y_org_train, y_org_test =\
    train_test_split(X_, X, Y, y_org, test_size=.5)

# first, do it with a standard CRF / SVM
pbl = GraphCRF(n_features=64, n_states=2, inference_method='lp')
svm = StructuredSVM(pbl, verbose=1, check_constraints=True, C=10000, n_jobs=1,
                    batch_size=-1)

svm.fit(X_train_, y_train)
y_pred = np.vstack(svm.predict(X_test_))
print("Score with pystruct crf svm: %f" % np.mean(y_pred == y_test))
print(svm.score(X_train_, y_train))
print(svm.score(X_test_, y_test))

# now with latent CRF SVM
latent_pbl = LatentGraphCRF(n_features=64, n_labels=2, n_states_per_label=5,
                            inference_method='dai')
latent_svm = LatentSSVM(latent_pbl, verbose=2, check_constraints=True, C=100,
                        n_jobs=1, batch_size=-1, tol=.1, latent_iter=2)
latent_svm.fit(X_train_, y_train)
print(latent_svm.score(X_train_, y_train))
print(latent_svm.score(X_test_, y_test))
