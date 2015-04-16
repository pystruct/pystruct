from os.path import dirname, join
import sys

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np


def _safe_unpickle(file_name):
    with open(file_name, "rb") as data_file:
        if sys.version_info >= (3, 0):
            # python3 unpickling of python2 unicode
            data = pickle.load(data_file, encoding="latin1")
        else:
            data = pickle.load(data_file)
    return data


def load_letters():
    """Load the OCR letters dataset.

    This is a chain classification task.
    Each example consists of a word, segmented into letters.
    The first letter of each word is ommited from the data,
    as it was a capital letter (in contrast to all other letters).


    References
    ----------
    http://papers.nips.cc/paper/2397-max-margin-markov-networks.pdf
    http://groups.csail.mit.edu/sls/archives/root/publications/1995/Kassel%20Thesis.pdf
    http://www.seas.upenn.edu/~taskar/ocr/
    """
    module_path = dirname(__file__)
    data = _safe_unpickle(join(module_path, 'letters.pickle'))
    # we add an easy to use image representation:
    data['images'] = [np.hstack([l.reshape(16, 8) for l in word])
                      for word in data['data']]
    return data


def load_scene():
    """Load the scene multi-label dataset.

    This is a benchmark multilabel dataset.
    n_classes = 6
    n_fetures = 294
    n_samples_test = 1196
    n_samples_train = 1211

    References
    ----------
    Matthew R. Boutell, Jiebo Luo, Xipeng Shen, and Christopher M. Brown.
    Learning multi-label scene classification.
    """
    module_path = dirname(__file__)
    return _safe_unpickle(join(module_path, 'scene.pickle'))


def load_snakes():
    """Load the synthetic snake datasets.

    Taken from:
    Nowozin, S., Rother, C., Bagon, S., Sharp, T., Yao, B., & Kohli, P.
    Decision Tree Fields, ICCV 2011

    This is a 2d grid labeling task where conditinal pairwise interactions are
    important.
    See the reference for an explanation.
    """

    module_path = dirname(__file__)
    return _safe_unpickle(join(module_path, 'snakes.pickle'))
