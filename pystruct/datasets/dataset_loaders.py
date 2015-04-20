from os.path import dirname, join

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np


def load_letters():
    """Load the OCR letters dataset.

    This is a chain classification task.
    Each example consists of a word, segmented into letters.
    The first letter of each word is ommited from the data,
    as it was a capital letter (in contrast to all other letters).
    """
    module_path = dirname(__file__)
    data_file = open(join(module_path, 'letters.pickle'), 'rb')
    data = pickle.load(data_file)
    # we add an easy to use image representation:
    data['images'] = [np.hstack([l.reshape(16, 8) for l in word])
                      for word in data['data']]
    return data


def load_scene():
    """Load the scene multi-label dataset.

    This is a benchmark multilabel dataset.
    """
    module_path = dirname(__file__)
    data_file = open(join(module_path, 'scene.pickle'), 'rb')
    return pickle.load(data_file)


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
    data_file = open(join(module_path, 'snakes.pickle'), 'rb')
    return pickle.load(data_file)
