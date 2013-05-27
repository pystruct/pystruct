"""
==========
Latent SVM
==========
A Latent CRF with one node is the same as a latent multiclass SVM
Using the latent variables, we can learn non-linear models. This is the
same as a simple Latent SVM model. It would obviously be more effiencent
to implement a special case for Latent SVMs so we don't have to run an
inference procedure.
"""

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits

from pystruct.models import GraphCRF, LatentGraphCRF
from pystruct.learners import StructuredSVM, LatentSubgradientSSVM

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
svm = StructuredSVM(pbl, verbose=1, check_constraints=True, C=1000, n_jobs=1,
                    batch_size=-1)

svm.fit(X_train_, y_train)
y_pred = np.vstack(svm.predict(X_test_))
print("Score with pystruct crf svm: %f" % np.mean(y_pred == y_test))
print(svm.score(X_train_, y_train))
print(svm.score(X_test_, y_test))

# now with latent CRF SVM
latent_pbl = LatentGraphCRF(n_features=64, n_labels=2, n_states_per_label=5,
                            inference_method='dai')
latent_svm = LatentSubgradientSSVM(model=latent_pbl, max_iter=5000, C=1,
                                   verbose=2, n_jobs=1, learning_rate=0.1,
                                   show_loss_every=10, momentum=0.0,
                                   decay_exponent=0.5)
#latent_svm = LatentSSVM(latent_pbl, verbose=2, check_constraints=True, C=100,
                        #n_jobs=1, batch_size=-1, tol=.1, latent_iter=2)
latent_svm.fit(X_train_, y_train)
print(latent_svm.score(X_train_, y_train))
print(latent_svm.score(X_test_, y_test))

h_pred = np.hstack(latent_svm.predict_latent(X_test_))
print("Latent class counts: %s" % repr(np.bincount(h_pred)))
