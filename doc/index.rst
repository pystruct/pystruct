PyStruct - Structured Learning in Python
========================================
PyStruct aims at being an easy-to-use structured learning and prediction library.
Currently it implements only max-margin methods and a perceptron, but other algorithms
might follow.

The goal of PyStruct is to provide a well-documented tool for researchers as well as non-experts
to make use of structured prediction algorithms.
The design tries to stay as close as possible to the interface and conventions
of `scikit-learn <http://scikit-learn.org/dev>`_.

Currently the project is mostly maintained by Andreas Mueller, but contributions are very welcome.
I plan a stable release soon.

You can contact the authors either via the `mailing list <https://groups.google.com/forum/#!forum/pystruct>`_
or on `github <https://github.com/pystruct/pystruct>`_.


Introduction
=============
In order to do learning with PyStruct, you need to pick two or three things:
a model structure, a learning algorithm and optionally an inference algorithm.
By constructing a learner object from a model,
you get an object that can ``fit`` to training data
and can ``predict`` for unseen data (just like scikit-learn estimators).


Models, aka CRFs
----------------
These determine what your model looks like:
its graph structure and its loss function.

This is basically the part that you have to write yourself when using the Python
interface in SVM^struct. I am currently working only on pairwise models and there is
support for grids and general graphs. The SSVM implementations are agnostic
to the kind of model that is used, so you can easily extend the given models
to include higher-order potentials, for example.

Learning algorithms
-------------------
These set the parameters in a model based on training data.

Learners are agnostic of the kind of model that is used,
so all combinations are possible
and new models can be defined (to include, e.g., higher-order potentials)
without changing the learner.

The current learning algorithms implement max margin learning,
similar to SVM^struct.
There is a subgradient and a QP version.
It is possible to put positivity constraints on
certain weight. There is also a simple perceptron.


Inference solvers
-----------------
These perform inference: they run your model on data
in order to make predictions.

There are some options to use different solvers for inference. A linear
programming solver using GLPK is included. I have Python interfaces for several
other methods on github, including LibDAI, QPBO, AD3 and GCO (submodular graph
cuts).

This is where the heavy lifting is done and in some sense these backends are
interchangeable.

Currently I would recommend AD3 for very accurate solutions and QPBO for larger models.
The OneSlackSSVM includes an option (``switch_to``) to switch the solver to
a stronger or exact solver when no constraints can be found using the previous
solver (which should be a faster undergenerating solver, such as QPBO).

Examples
=========
See the example gallery:

.. toctree::

    auto_examples/index

Remarks
=======

For updates, read my blog at http://peekaboo-vision.blogspot.com

There are not publications yet that you can cite for this, I'm hoping there will be some in the future.

Btw: this is research with unit tests!

Installation
=============

There is no need to compile anything, this pure Python. (FIXME, Crammer-Singer has a cython part!)

There are quite a couple of requirements, though:

*   You need cvxopt for the cutting plane SVM solver and linear programming inference. By default I use the glpk solver for the LP, so you need that, too, if you want to use LP inference.

*   You need sklearn for some tidbits here and there, also I import joblib from sklearn.

*   For the other inference algorithms that are wrapped in the inference folder, you need the following of my repositories. You can just pick and choose from those, but lack of methods will make some tests fail.

    QPBO https://github.com/amueller/pyqpbo

    libdai https://github.com/amueller/daimrf

    AD3 https://github.com/amueller/AD3

.. include:: references.rst

