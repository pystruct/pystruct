TODO
================
* rename positivity_constraint to negativity_constraint - or flip some signs!
* implement customized kkt solver for inner qp (should we?)
* make tests faster:
    - use OneSlackSSVM more in tests
* constraint pruning in n-slack SSVM
* finish multilabel
* missing examples:
    * handwritten sequence classification example
    * more chain CRFs - POS tagging?
    * submodular CRFs - segmentation?
* make more examples plot examples
* replace pairwise tri code with scipy.spatial.squareform
* in subgradient ssvm: Decouple updates from shrinking w (aka commulative penalty).
* remove EdgeTypeGraphCRF
* allow for warm starts in inference during learning
