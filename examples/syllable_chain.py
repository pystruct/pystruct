import urllib2
import numpy as np

from sklearn.feature_extraction import FeatureHasher
from sklearn.cross_validation import train_test_split

from pystruct.models import ChainCRF
from pystruct.learners import SubgradientSSVM


NETTALK = ("http://archive.ics.uci.edu/ml/machine-learning-databases/"
           "undocumented/connectionist-bench/nettalk/nettalk.data")


def features(word):
    def end_features(curr_position):
        return ("c[-1]={}".format(word[curr_position - 1]),
                "c[+1]={}".format(word[curr_position]))

    def middle_features(curr_position):
        return ("c[-2]={}".format(word[curr_position - 2]),
                "c[+2]={}".format(word[curr_position + 1]),
                "c[-2:-1]={}".format(word[curr_position - 1:curr_position]),
                "c[1:2]={}".format(word[curr_position + 1:curr_position + 3]))

    return [end_features(pos) if pos == 1 or pos == len(word) - 1
            else end_features(pos) + middle_features(pos)
            for pos in xrange(1, len(word))]


def nettalk_syl_to_split(syl):
    syllables = [syl[k - 1] != '>' and syl[k] in ('>', '0', '1', '2')
                 for k in xrange(1, len(syl))]
    stress = [k == '1' for k in syl]
    return syllables, stress


def nettalk_line(line):
    try:
        word, phon, syl, cls = line.strip().split('\t')
        syllable, stress = nettalk_syl_to_split(syl)
    except ValueError:
        word, syllable, stress = "", [], []
    return features(word), syllable, stress


def numbered_nb(y):
    new_y = np.empty(len(y), dtype=np.int)
    last_split = -1
    for k, is_split in enumerate(y):
        if is_split:
            last_split = k
        new_y[k] = k - last_split
    return new_y


if __name__ == '__main__':
    url = urllib2.urlopen(NETTALK)
    for _ in xrange(10):  # skip header
        url.readline()
    lines = [nettalk_line(line) for line in url]
    url.close()

    X, y = zip(*((word, tag) for (word, tag, _) in lines if len(word)))
    hasher = FeatureHasher(input_type='string', n_features=2**10,
                           non_negative=True)
    X = np.array([hasher.transform(instance) for instance in X])
    y = np.array([numbered_nb(this_y) for this_y in y])

    # The random state ensures that all labels are in the train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2,
                                                        random_state=1)

    # train linear chain CRF
    model = ChainCRF(inference_method=('ad3', dict(branch_and_bound=True)))
    ssvm = SubgradientSSVM(model=model, verbose=1, C=100, max_iter=5)
    ssvm.fit(X_train, y_train)
    y_pred = ssvm.predict(X_test)
    score = np.mean([np.all((y_t == 0) == (y_p == 0))
                    for (y_t, y_p) in zip(y_test, y_pred)])
    print("Test score: {:2.2f}".format(ssvm.score(X_test, y_test)))
