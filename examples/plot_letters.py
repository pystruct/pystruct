import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

from pystruct.datasets import load_letters
from pystruct.models import ChainCRF
#from pystruct.learners import NSlackSSVM
from pystruct.learners import OneSlackSSVM
#from pystruct.learners import SubgradientSSVM

letters = load_letters()
X, y, folds = letters['data'], letters['labels'], letters['folds']
train_indices = np.where(folds == 1)[0]
test_indices = np.where(folds != 1)[0]
X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

model = ChainCRF(inference_method='unary')
#ssvm = NSlackSSVM(model=model, verbose=2, C=0.01, batch_size=1000,
                  #inactive_threshold=1e-3)
ssvm = OneSlackSSVM(model=model, verbose=2, C=0.01)
#ssvm = SubgradientSSVM(model=model, verbose=2, C=0.01, learning_rate=.0001)
ssvm.fit(X_train, y_train)

print(ssvm.score(X_train, y_train))
print(ssvm.score(X_test, y_test))

#from IPython.core.debugger import Tracer
#Tracer()()

#svm = LinearSVC(dual=False, C=1)
#svm.fit(np.vstack(X_train), np.hstack(y_train))
#print(svm.score(np.vstack(X_train), np.hstack(y_train)))
#print(svm.score(np.vstack(X_test), np.hstack(y_test)))


#plt.matshow(np.hstack([x.reshape(16, 8) for x in svm.coef_]),
            #cmap=plt.cmap.Grey)

#plt.matshow(confusion_matrix(np.hstack(y_test),
                             #svm.predict(np.vstack(X_test))))
#abc = "abcdefghijklmnopqrstuvwxyz"
#plt.xticks(np.arange(26), abc)
#plt.show()
