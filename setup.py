from setuptools import setup
from setuptools.extension import Extension

import os

if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')


class MyExt(Extension, object):
    """
    Postpone np.get_include() until extensions are being built
    https://stackoverflow.com/a/54128391
    """

    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super(MyExt, self).__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy as np
        return self.__include_dirs + [np.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self.__include_dirs = dirs


ext_modules = [
    MyExt("pystruct.models.utils", ["src/utils.c"]),
    MyExt("pystruct.inference._viterbi", ["pystruct/inference/_viterbi.c"]),
]

setup(name="pystruct",
      version="0.3.2",
      setup_requires=["cython", "numpy"],
      install_requires=["ad3", "numpy"],
      packages=['pystruct', 'pystruct.learners', 'pystruct.inference',
                'pystruct.models', 'pystruct.utils', 'pystruct.datasets',
                'pystruct.tests', 'pystruct.tests.test_learners',
                'pystruct.tests.test_models', 'pystruct.tests.test_inference',
                'pystruct.tests.test_utils'],
      include_package_data=True,
      description="Structured Learning and Prediction in Python",
      author="Andreas Mueller",
      author_email="t3kcit@gmail.com",
      url="http://pystruct.github.io",
      license="BSD 2-clause",
      use_2to3=True,
      ext_modules=ext_modules,
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.6',
                   ],
      )
