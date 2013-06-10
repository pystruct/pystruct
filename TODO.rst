TODO
================
* in subgradient ssvm: Decouple updates from shrinking w (aka commulative penalty).
* rename positivity_constraint to negativity_constraint - or flip some signs!
* implement customized kkt solver for inner qp (should we?)
* use OneSlackSSVM more in tests -> faster tests
* constraint pruning in n-slack SSVM
* organize tests
* finish multilabel
* rename StructuredSVM to NSlackSSVM - or make a superclass from one-slack and n-slack?
* missing examples:
    * handwritten sequence classification example
    * more chain CRFs - POS tagging?
    * submodular CRFs - segmentation?
* make more examples plot examples
* replace pairwise tri code with scipy.spatial.squareform
