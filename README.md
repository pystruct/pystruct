PyStruct
========

This is my humble structural SVM and CRF implementation.
I use it for my research and hope you find it helpful.
Be aware that it might change drastically.

Content
-------
There are three basic parts to the implementation:

- Structural SVMs
    These implement max margin learning, similar to SVM^struct.
    There is a subgradient and a QP version. 
    They are not particularly optimized but at this part is usually not the
    bottleneck in structured learning, so the is not really an issue. It is
    possible to put positivity constraints on certain weight.

- CRFs / Problems
    These know about the structure of the problem, the loss and the inference.
    This is basically the part that you have to write yourself when using the
    Python interface in SVM^struct.
    I am only working on pairwise models and there is support for grids and
    general graphs. I am mostly working on the grids at the moment.

- Inference Solvers
    There are some options to use differnent solvers for inference.
    A linear programming solver using GLPK is included.
    I have Python interfaces for several other methods on github,
    including LibDAI, QPBO, AD3 and GCO (submodular graph cuts).


There is also some stuff on latent SVMs here that is my current research and
I'd ask you not to steal it ;)

For updates, read my blog at http://peekaboo-vision.blogspot.com
