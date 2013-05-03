.. pystruct documentation master file, created by
   sphinx-quickstart on Fri May  3 17:14:50 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

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

Know about the problem.

These know about the structure of the problem, the loss and the inference. This
is basically the part that you have to write yourself when using the Python
interface in SVM^struct. I am only working on pairwise models and there is
support for grids and general graphs. I am mostly working on the grids at the
moment.


Inference Solvers
==================
Doe the inference.

There are some options to use different solvers for inference. A linear
programming solver using GLPK is included. I have Python interfaces for several
other methods on github, including LibDAI, QPBO, AD3 and GCO (submodular graph
cuts).

This is where the heavy lifting is done and in some sense these backends are
exchangeable. I'm hoping to unify stuff a bit more here.

Remarks
=======

There is also some stuff on latent SVMs here that is my current research and I'd ask you not to steal it ;)

For updates, read my blog at http://peekaboo-vision.blogspot.com

There are not publications yet that you can cite for this, I'm hoping there will be some in the future.

Btw: this is research with unit tests!
Installation

There is no need to compile anything, this pure Python. (FIXME, Crammer-Singer has a cython part!)

There are quite a couple of requirements, though:

*   You need cvxopt for the cutting plane SVM solver and linear programming inference. By default I use the glpk solver for the LP, so you need that, too, if you want to use LP inference.

*   You need sklearn for some tidbits here and there, also I import joblib from sklearn.

*   For the other inference algorithms that are wrapped in the inference folder, you need the following of my repositories. You can just pick and choose from those, but lack of methods will make some tests fail.

    QPBO https://github.com/amueller/pyqpbo

    libdai https://github.com/amueller/daimrf

    AD3 https://github.com/amueller/AD3



.. toctree::
   :maxdepth: 2

Learning
==========
This module contains algorithms for solving the structured learning problem.
Most are based on structured support vector machines.

Currently, I advise to use the OneSlackSSVM, which solves the QP using CVXOPT.
SubgradientSSVM is a very simple implementation, that also might be interesting.

StructuredSVM is the n-slack formulation of the QP and should work reliably,
but is not as optimized as OneSlackSSVM.
The rest is experimental / for testing.

.. automodule:: pystruct.learners
   :no-members:
   :no-inherited-members:

.. currentmodule:: pystruct

.. autosummary::
   :toctree: generated/
   :template: class.rst

    learners.OneSlackSSVM
    learners.StructuredSVM
    learners.SubgradientSSVM
    learners.StructuredPerceptron
    learners.LatentSSVM
    learners.LatentSubgradientSSVM
    learners.PrimalDSStructuredSVM

Models
========
This module contains problem formulations for several settings.
They provide the glue between the learning algorithm and the data (and inference).
The BinarySVMProblem implements a standard SVM, the CrammerSingerSVMProblem a multi-class SVM
- which is surprisingly efficient and sometimes comparable to LibLinear Crammer-Singer Implementation.

GraphCRF implements a simple pairwise model for arbitrary graphs, while EdgeFeatureGraphCRF allows
for arbitrary features for each edge, symmetric, assymmetric and arbitrary potentials.

GridCRF is a convenience class for grid graphs.

.. automodule:: pystruct.models
   :no-members:
   :no-inherited-members:

.. currentmodule:: pystruct

.. autosummary::
   :toctree: generated/
   :template: class.rst

    models.BinarySVMProblem
    models.CrammerSingerSVMProblem
    models.GraphCRF
    models.EdgeFeatureGraphCRF
    models.LatentGraphCRF
    models.GridCRF
    models.DirectionalGridCRF


Inference
===========

.. automodule:: pystruct.inference
   :no-members:
   :no-inherited-members:

.. currentmodule:: pystruct

.. autosummary::
   :toctree: generated/
   :template: function.rst

   inference.inference_dispatch
   inference.inference_qpbo
   inference.inference_dai
   inference.inference_lp
   inference.inference_ad3

Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


