.. _installation:

Installation
=============
To install pystruct, you need cvxopt, cython and scikit-learn (which requires numpy and scipy).

The easiest way to install pystruct is using pip::

    pip install pystruct

This will also install the additional inference package ad3.

Installation instructions for the requirements are below.

Linux (Ubuntu)
--------------
The easiest way to get all requirements is via the package manager, that is apt on Ubuntu and Debian::

    sudo apt-get install build-essential python-dev python-setuptools python-numpy \\
        python-scipy libatlas-dev libatlas3gf-base python-cvxopt

To install the current versions of scikit-learn and pystruct, you can use pip::

    pip install --user --upgrade scikit-learn pystruct

OS X & Windows
---------------
Follow instructions on the `scikit-learn website <http://scikit-learn.org/dev/install.html>`_ and
then `install CVXOPT <http://cvxopt.org/install/>`_.
Finally, you can install pystruct simply using::

    pip install --user --upgrade pystruct


Alternative: Anaconda
---------------------
In particular for OS X and Windows, an alternative is to use the `Anaconda Python <https://store.continuum.io/cshop/anaconda/>`_ distribution.
The anaconda environment comes with its own Python and a package manager named conda.
You can install cvxopt using the conda package manager::

    conda install cvxopt

And then install pystruct::

    pip install --user --upgrade pystruct

Additional inference packages
=============================
While PyStruct implements some simple inference algorithms, these are not as optimized as other available code.
Therefore it is recommended to install additional inference packages.
By default PyStruct will also install the AD3 package, which contains a high-quality solver
that can be chosen via ``inference_method='ad3'``.
Another solver that is helpful for highly connected graphs like grid-graphs is QPBO, which
can be installed via the pyqpbo package::

    pip install --user pyqpbo

Unfortunately QPBO might not compile with newer C compilers, so we decided to not make it a dependency.

.. currentmodule:: pystruct.inference

There is a very high quality collection of inference algorithms in
the `OpenGM <http://ipa.iwr.uni-heidelberg.de/jkappes/opengm2/>`_ library, which 
is highly recommended. The algorithms in OpenGM can be chosen by specifying
``inference_algorithm=('ogm', {'alg': ALGORITHM})`` where ALGORITHM can be a
wide variety of algorithms, including dynamic programming, TRWS, graph cuts and
many more, see :func:`inference_ogm`.

In particular for tree-structured (not chain) models, the implementation of dynamic
programming max-product belief propagation in OpenGM is much faster than the
one in PyStruct.
