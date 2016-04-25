from setuptools import setup
from setuptools.extension import Extension
import numpy as np

import os

if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

include_dirs = [np.get_include()]

setup(name="pystruct",
      version="0.2.5",
      install_requires=["ad3"],
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
      ext_modules=[Extension("pystruct.models.utils", ["src/utils.c"],
                             include_dirs=include_dirs),
                   Extension("pystruct.inference._viterbi",
                             ["pystruct/inference/_viterbi.c"],
                             include_dirs=include_dirs)],
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
                   'Programming Language :: Python :: 2.6',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.3',
                   ],
      )
