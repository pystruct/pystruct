from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(name="pystruct",
      version="0.0.1",
      cmdclass={'build_ext': build_ext},
      ext_modules=[Extension("pystruct.models.utils", ["src/utils.pyx"])])
