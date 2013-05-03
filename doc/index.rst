.. pystruct documentation master file, created by
   sphinx-quickstart on Fri May  3 17:14:50 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pystruct's documentation!
====================================

Contents:

.. toctree::
   :maxdepth: 2


.. automodule:: pystruct.learners
   :no-members:
   :no-inherited-members:

.. currentmodule:: pystruct

.. autosummary::
   :toctree: generated/
   :template: class.rst

    learners.OneSlackSSVM
    learners.StructurdSVM
    learners.SubgradientSSVM
    learners.StructuredPerceptron
    learners.LatentSSVM
    learners.LatentSubgradientSSVM
    learners.PrimalDSStructuredSVM


.. automodule:: pystruct.problems
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

    problems.GridCRF
    problems.GraphCRF
    problems.DirectionalGraphCRF
    problems.BinarySVMProblem
    problems.CrammerSingerSVMProblem
    problems.EdgeFeatureGraphCRF
    problems.LatentGraphCRF


.. automodule:: pystruct.inference
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   inference.inference_dispatch
   inference.inference_qpbo
   inference.inference_dai
   inference.inference_lp
   inference.inference_ad3
   inference.inference_ogm

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


