import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

from pystruct.datasets import load_letters
from pystruct.models import ChainCRF
from pystruct.learners import OneSlackSSVM

letters = load_letters()
X, y, folds = letters['data'], letters['labels'], letters['folds']
# we convert the lists to object arrays, as that makes slicing much more
# convenient
X, y = np.array(X), np.array(y)
X_train, X_test = X[folds == 1], X[folds != 1]
y_train, y_test = y[folds == 1], y[folds != 1]

#model = ChainCRF(inference_method='qpbo')
#ssvm = OneSlackSSVM(model=model, verbose=2, C=0.1, inference_cache=50)
#ssvm.fit(X_train, y_train)

#print(ssvm.score(X_train, y_train))
#print(ssvm.score(X_test, y_test))


svm = LinearSVC(dual=False, C=1)
svm.fit(np.vstack(X_train), np.hstack(y_train))
print(svm.score(np.vstack(X_train), np.hstack(y_train)))
print(svm.score(np.vstack(X_test), np.hstack(y_test)))


plt.matshow(np.hstack([x.reshape(16, 8) for x in svm.coef_]),
            cmap=plt.cm.Greys)

plt.matshow(confusion_matrix(np.hstack(y_test),
                             svm.predict(np.vstack(X_test))))
abc = "abcdefghijklmnopqrstuvwxyz"
plt.xticks(np.arange(26), abc)
plt.show()
