Welcome to pystruct's documentation!
====================================

This is my humble structural SVM and CRF implementation. I use it for my research and hope you find it helpful. Be aware that it might change drastically.

There are three basic parts to the implementation.


Structural SVMs
================
Know about learning.

These implement max margin learning, similar to SVM^struct. There is a
subgradient and a QP version. They are not particularly optimized but at this
part is usually not the bottleneck in structured learning, so the is not really
an issue. It is possible to put positivity constraints on certain weight. There
is also a simple perceptron.

CRFs aka Models
==================

Know about the model formulation.

These know about the structure of the model, the loss and the inference. This
is basically the part that you have to write yourself when using the Python
interface in SVM^struct. I am only working on pairwise models and there is
support for grids and general graphs. I am mostly working on the grids at the
moment.


Inference Solvers
==================
Do the inference.

There are some options to use different solvers for inference. A linear
programming solver using GLPK is included. I have Python interfaces for several
other methods on github, including LibDAI, QPBO, AD3 and GCO (submodular graph
cuts).

This is where the heavy lifting is done and in some sense these backends are
interchangeable.

Currently I would recommend AD3 for very accurate solutions and QPBO for larger models.
The OneSlackSSVM includes an option (``switch_to_ad3``) to switch the solver to AD3 when no
constraints can be found using the previous solver (which should be a faster undergenerating solver, such as QPBO).

Examples
=========
See the example gallery:

.. toctree::

    auto_examples/index

Remarks
=======

There is also some stuff on latent SVMs here that is my current research and I'd ask you not to steal it ;)

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

