import cPickle
from os.path import dirname
from os.path import join

import numpy as np


def load_letters():
    """Load the OCR letters dataset.

    This is a chain classification task.
    Each example consists of a word, segmented into letters.
    The first letter of each word is ommited from the data,
    as it was a capital letter (in contrast to all other letters).
    """
    module_path = dirname(__file__)
    data_file = open(join(module_path, 'letters.pickle'),'rb')
    data = cPickle.load(data_file)
    # we add an easy to use image representation:
    data['images'] = [np.hstack([l.reshape(16, 8) for l in word])
                      for word in data['data']]
    return data


def load_scene():
    module_path = dirname(__file__)
    data_file = open(join(module_path, 'scene.pickle'))
    return cPickle.load(data_file)


def load_snakes():
    module_path = dirname(__file__)
    data_file = open(join(module_path, 'snakes.pickle'))
    return cPickle.load(data_file)
