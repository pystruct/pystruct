from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

src = "../ad3/"

#files = ['GCoptimization.cpp', 'graph.cpp', 'LinkedBlockList.cpp',
    #'maxflow.cpp']

#files = [gco_directory + f for f in files]
#files.insert(0, "gco_python.pyx")

setup(cmdclass={'build_ext': build_ext}, ext_modules=[
    Extension("ad3", ["factor_graph.pyx"], language="c++",
              include_dirs=["../"], library_dirs=[src], libraries=["ad3"])])
