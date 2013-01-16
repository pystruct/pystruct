# a Latent CRF with one node is the same as a latent multiclass SVM
# Using the latent variables, we can learn non-linear problems

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

from pystruct.problems import GraphCRF, LatentGraphCRF
from pystruct.learners import StructuredSVM, LatentSSVM

# generate a binary, non-linear dataset
X = np.random.randn(300, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(np.int)

# make each example into a tuple of a single feature vector and an empty edge
# list
X_ = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]
Y = y.reshape(-1, 1)

X_train_, X_test_, X_train, X_test, y_train, y_test = \
    train_test_split(X_, X, Y, test_size=.5)

# first, do it with a standard CRF / SVM
pbl = GraphCRF(n_features=2, n_states=2, inference_method='lp')
svm = StructuredSVM(pbl, verbose=1, check_constraints=True, C=100, n_jobs=1)

svm.fit(X_train_, y_train)
y_pred = np.vstack(svm.predict(X_test_))
print("Score with pystruct crf svm: %f" % np.mean(y_pred == y_test))

# now with latent CRF SVM
latent_pbl = LatentGraphCRF(n_features=2, n_labels=2, n_states_per_label=2,
                            inference_method='lp')
latent_svm = LatentSSVM(latent_pbl, verbose=1, check_constraints=True, C=100,
                        n_jobs=1, plot=False)
latent_svm.fit(X_train_, y_train,
               H_init=np.random.randint(2, size=y_train.shape) + 2 * y_train)
y_pred_latent = np.vstack(latent_svm.predict(X_test_))

# plot the results
fig, axes = plt.subplots(1, 5, figsize=(12, 4))
cm = plt.cm.Paired
axes[0].set_title("Ground truth")
axes[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap=cm)
axes[1].set_title("SVM prediction")
axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_pred, s=50, cmap=cm)
axes[2].set_title("Latent SVM prediction")
axes[2].scatter(X_test[:, 0], X_test[:, 1], c=y_pred_latent // 2, s=50,
                cmap=cm)
axes[3].set_title("Latent states")
axes[3].scatter(X_test[:, 0], X_test[:, 1], c=y_pred_latent, s=50, cmap=cm)
axes[4].set_title("Initial states")
axes[4].scatter(X_train[:, 0], X_train[:, 1], c=np.vstack(latent_svm.H_init_),
                s=50, cmap=cm)
for ax in axes:
    ax.set_xticks(())
    ax.set_yticks(())
plt.show()
