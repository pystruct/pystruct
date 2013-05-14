TODO
================
* setup.py / makefile in main director!
* in subgradient ssvm: Decouple updates from shrinking w (aka commulative penalty).
* rename positivity_constraint to negativity_constraint - or flip some signs!
* loss computation in ssvm.py is weird. It should be acuraccy, decoupled from the loss we are optimizing.
* implement customized kkt solver for inner qp
* use OneSlackSSVM more in tests -> faster tests
* constraint pruning in n-slack SSVM
* organize tests
* finish multilabel
* handwritten sequence classification example
* rename StructuredSVM to NSlackSSVM - or make a superclass from one-slack and n-slack?
