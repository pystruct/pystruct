.. toctree::
   :maxdepth: 2

.. _learning:

Learning
==========
This module contains algorithms for solving the structured learning model.
Most are based on structured support vector machines.

Currently, I advise to use the OneSlackSSVM, which solves the QP using CVXOPT.
SubgradientSSVM is a very simple implementation, that also might be interesting.

NSlackSSVM is the n-slack formulation of the QP and should work reliably,
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
    learners.NSlackSSVM
    learners.SubgradientSSVM
    learners.StructuredPerceptron
    learners.LatentSSVM
    learners.SubgradientLatentSSVM
    learners.PrimalDSStructuredSVM
    learners.FrankWolfeSSVM

.. _models:

Models
========
This module contains model formulations for several settings. They provide the
glue between the learning algorithm and the data (and inference).

There are two main classes of models, conditional random field models (CRFs)
and classification models (Clfs).

The BinaryClf implements a standard binary classifier, the MultiClassClf a
linear multi-class classifier. Together with a max-margin learner, these
produce standard binary SVMs and Crammer-Singer multi-class SVMs. MultiLabelClf
implements a multi label model with different possible pairwise interactions.

GraphCRF implements a simple pairwise model for arbitrary graphs, while
EdgeFeatureGraphCRF allows for arbitrary features for each edge, symmetric,
assymmetric and arbitrary potentials.

GridCRF is a convenience class for grid graphs.

.. automodule:: pystruct.models
   :no-members:
   :no-inherited-members:

.. currentmodule:: pystruct

Classifiers
-----------

.. autosummary::
   :toctree: generated/
   :template: class.rst

    models.BinaryClf
    models.MultiClassClf
    models.MultiLabelClf

Conditional Random Fields
-------------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

    models.GraphCRF
    models.EdgeFeatureGraphCRF
    models.LatentGraphCRF
    models.LatentNodeCRF
    models.ChainCRF
    models.GridCRF
    models.DirectionalGridCRF

.. _inference:

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
   inference.inference_lp
   inference.inference_ad3
   inference.inference_ogm

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

plot_learning
-------------

.. automodule:: pystruct.plot_learning
   :no-members:
   :no-inherited-members:

.. currentmodule:: pystruct

.. autosummary::
   :toctree: generated/
   :template: function.rst

    plot_learning.plot_learning
