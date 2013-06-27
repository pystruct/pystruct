# encoding: utf8
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from pystruct.models import ChainCRF
from pystruct.learners import StructuredPerceptron

def word_to_features(word, size=2):
    instance = []
    for k, char_ in enumerate(word):
        if k == 0:
            continue
        features = {str(-1 - label): letter for label, letter in enumerate(word[k - size + 1:k + 1])}
        features.update({str(1 + label): letter for label, letter in enumerate(word[k:k+size])})
        instance.append(features)

    return instance

x_s = [u'carte', u'împarte', u'arte', u'curte']
y = [[0, 0, 1, 0], [0, 1, 0, 0, 1, 0], [0, 1, 0], [0, 0, 1, 0]]
y = [np.array(k) for k in y]

x_dict = [word_to_features(w) for w in x_s]
vect = DictVectorizer(sparse=False).fit([d for item in x_dict for d in item])
x = np.array([vect.transform(item) for item in x_dict])


perc = StructuredPerceptron(model=ChainCRF(n_states=2, n_features=x[0].shape[1]), max_iter=5)
perc.fit(x[:2], y[:2])
print perc.score(x[2:], y[2:])
