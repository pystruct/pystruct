PyStruct - Structured Learning in Python
========================================
PyStruct aims at being an easy-to-use structured learning and prediction library.
Currently it implements only max-margin methods and a perceptron, but other algorithms
might follow. The learning algorithms implemented in PyStruct have various names,
which are often used loosely or differently in different communities.
Common names are conditional random fields (CRFs), maximum-margin Markov
random fields (M3N) or structural support vector machines.

If you are new to structured learning,
have a look at :ref:`intro`.

The goal of PyStruct is to provide a well-documented tool for researchers as well as non-experts
to make use of structured prediction algorithms.
The design tries to stay as close as possible to the interface and conventions
of `scikit-learn <http://scikit-learn.org/dev>`_.

The current version is PyStruct 0.2.2 which you can install via pip:

    pip install pystruct

Detailed installation instructions can be found under :ref:`installation`.

Starting with this first stable release, PyStruct will remain
stable with respect to API and will provide backward compatibility.

You can contact the authors either via the `mailing list <https://groups.google.com/forum/#!forum/pystruct>`_
or on `github <https://github.com/pystruct/pystruct>`_.

Citing
======
If you find PyStruct helpful, please cite `our paper <http://jmlr.org/papers/volume15/mueller14a/mueller14a.pdf>`_:
    
    | Andreas C. Mueller, Sven Behnke
    | PyStruct - Structured prediction in Python
    | Journal of machine learning, 2014
    | `bibtex here <http://jmlr.org/papers/v15/mueller14a.bib>`_


Introduction
=============
In order to do learning with PyStruct, you need to pick two or three things:
a model structure, a learning algorithm and optionally an inference algorithm.
By constructing a learner object from a model, you get an object that can
``fit`` to training data and can ``predict`` for unseen data (just like
scikit-learn estimators).


Models, aka CRFs
----------------
These determine what your model looks like:
its graph structure and its loss function.
There are several ready-to-use classes, for example for multi-label
classification, chain CRFs and more complex models. You can find a
full list in the :ref:`models` section of the references

Learning algorithms
-------------------
These set the parameters in a model based on training data.

Learners are agnostic of the kind of model that is used, so all combinations
are possible and new models can be defined (to include, e.g., higher-order
potentials) without changing the learner.

The current learning algorithms implement max margin learning and
a perceptron. See the :ref:`learning` section of the references.


Inference solvers
-----------------
These perform inference: they run your model on data
in order to make predictions.

There are some options to use different solvers for inference. A linear
programming solver using cvxopt is included. I have Python interfaces for several
other methods on github, including LibDAI, QPBO, AD3.

This is where the heavy lifting is done and in some sense these backends are
interchangeable.

Currently I would recommend AD3 for very accurate solutions and QPBO for larger models.
The the cutting plane solvers include an option (``switch_to``) to switch the solver to
a stronger or exact solver when no constraints can be found using the previous
solver (which should be a faster undergenerating solver, such as QPBO).

.. toctree::
    :hidden:

    auto_examples/index
    references.rst
    intro.rst
    installation.rst
