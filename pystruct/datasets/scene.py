import cPickle
from os.path import dirname
from os.path import join


def load_scene():
    module_path = dirname(__file__)
    data_file = open(join(module_path, 'scene.pickle'))
    return cPickle.load(data_file)
