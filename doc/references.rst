.. toctree::
   :maxdepth: 2

Learning
==========
This module contains algorithms for solving the structured learning model.
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
This module contains model formulations for several settings.
They provide the glue between the learning algorithm and the data (and inference).
The BinarySVMModel implements a standard SVM, the CrammerSingerSVMModel a multi-class SVM
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

    models.BinarySVMModel
    models.CrammerSingerSVMModel
    models.GraphCRF
    models.EdgeFeatureGraphCRF
    models.LatentGraphCRF
    models.LatentNodeCRF
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

Utilities
===========
.. automodule:: pystruct.utils
   :no-members:
   :no-inherited-members:

.. currentmodule:: pystruct

.. autosummary::
   :toctree: generated/
   :template: class_with_call.rst

    utils.SaveLogger

