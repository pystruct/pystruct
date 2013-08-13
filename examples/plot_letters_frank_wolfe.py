"""
===============================
OCR Letter sequence recognition
===============================
This example illustrates the use of a chain CRF for optical character
recognition. The example is taken from Taskar et al "Max-margin markov random
fields".

Each example consists of a handwritten word, that was presegmented into
characters.  Each character is represented as a 16x8 binary image. The task is
to classify the image into one of the 26 characters a-z. The first letter of
every word was ommited as it was capitalized and the task does only consider
small caps letters.

We compare classification using a standard linear SVM that classifies
each letter individually with a chain CRF that can exploit correlations
between neighboring letters (the correlation is particularly strong
as the same words are used during training and testsing).

The first figures shows the segmented letters of four words from the test set.
In set are the ground truth (green), the prediction using SVM (blue) and the
prediction using a chain CRF (red).

The second figure shows the pairwise potentials learned by the chain CRF.
The strongest patterns are "y after l" and "n after i".

There are obvious extensions that both methods could benefit from, such as
window features or non-linear kernels. This example is more meant to give a
demonstration of the CRF than to show its superiority.
"""
import numpy as np
#import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
#from sklearn.metrics import confusion_matrix

from pystruct.datasets import load_letters
from pystruct.models import ChainCRF
from pystruct.learners import OneSlackSSVM
from pystruct.learners import FrankWolfeSSVM
abc = "abcdefghijklmnopqrstuvwxyz"
from pystruct.models import MultiClassClf

letters = load_letters()
X, y, folds = letters['data'], letters['labels'], letters['folds']
# we convert the lists to object arrays, as that makes slicing much more
# convenient
for i, x in enumerate(X):
    X[i] = np.hstack([x, np.ones((x.shape[0], 1))])

cur_fold = 1
X, y = np.array(X), np.array(y)
X_train, X_test = X[folds == cur_fold], X[folds != cur_fold]
y_train, y_test = y[folds == cur_fold], y[folds != cur_fold]

# Train linear SVM
svm = LinearSVC(dual=False, C=.1)
# flatten input
svm.fit(np.vstack(X_train), np.hstack(y_train))
print("Test score with linear SVM: %f" % svm.score(np.vstack(X_train),
                                                   np.hstack(y_train)))

if False:
    # multi class SVM
    svm2 = FrankWolfeSSVM(MultiClassClf(n_features=129, n_classes=26), C=100, max_iter=200, line_search=True,
                          batch_mode=False, dual_check_every=np.vstack(X_train).shape[0])
    svm2.fit(np.vstack(X_train), np.hstack(y_train))
    print svm2.score(np.vstack(X_train), np.hstack(y_train))


# Train linear chain CRF
if False:
    ssvm = OneSlackSSVM(ChainCRF(inference_method='dai'), C=.1, inference_cache=50, tol=0.1, verbose=1)
    ssvm.fit(X_train, y_train)
    print("Test score with chain CRF: %f" % ssvm.score(X_test, y_test))


if False:
    svm1 = FrankWolfeSSVM(ChainCRF(inference_method='dai'), C=100, max_iter=50, line_search=True, batch_mode=True)
    svm1.fit(X_train, y_train)
    print("%f" % svm1.score(X_train, y_train))

if True:
    svm2 = FrankWolfeSSVM(ChainCRF(), C=.1, max_iter=500, line_search=True, batch_mode=False,
                          dual_check_every=5000, verbose=1)
    svm2.fit(X_train, y_train)
    print(" %f" % svm2.score(X_train, y_train))
    print(" %f" % svm2.score(X_test, y_test))

# plot some word sequenced
# n_words = 4
# rnd = np.random.RandomState(1)
# selected = rnd.randint(len(y_test), size=n_words)
# max_word_len = max([len(y) for y in y_test[selected]])
# fig, axes = plt.subplots(n_words, max_word_len, figsize=(10, 10))
# fig.subplots_adjust(wspace=0)
# for ind, axes_row in zip(selected, axes):
#     y_pred_svm = svm.predict(X_test[ind])
#     y_pred_chain = ssvm.predict([X_test[ind]])[0]
#     for i, (a, image, y_true, y_svm, y_chain) in enumerate(
#             zip(axes_row, X_test[ind], y_test[ind], y_pred_svm, y_pred_chain)):
#         a.matshow(image.reshape(16, 8), cmap=plt.cm.Greys)
#         a.text(0, 3, abc[y_true], color="#00AA00", size=25)
#         a.text(0, 14, abc[y_svm], color="#5555FF", size=25)
#         a.text(5, 14, abc[y_chain], color="#FF5555", size=25)
#         a.set_xticks(())
#         a.set_yticks(())
#     for ii in xrange(i + 1, max_word_len):
#         axes_row[ii].set_visible(False)
# 
# plt.matshow(ssvm.w[26 * 8 * 16:].reshape(26, 26))
# plt.title("Transition parameters of the chain CRF.")
# plt.xticks(np.arange(25), abc)
# plt.yticks(np.arange(25), abc)
# plt.show()
