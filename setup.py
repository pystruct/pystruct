from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import os

if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

setup(name="pystruct",
      version="0.1",
      packages=['pystruct', 'pystruct.learners', 'pystruct.inference',
                'pystruct.models', 'pystruct.utils', 'pystruct.datasets'],
      description="Structured Learning and Prediction in Python",
      author="Andreas Mueller",
      author_email="t3kcit@gmail.com",
      url="http://pystruct.github.io",
      license="BSD 2-clause",
      cmdclass={'build_ext': build_ext},
      ext_modules=[Extension("pystruct.models.utils", ["src/utils.pyx"])],
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
