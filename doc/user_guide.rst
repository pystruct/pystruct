Multi-class SVM
===============
A precursor for structured SVMs was the mult-class SVM by Crammer and Singer.
While in practice it is often faster to use an One-vs-Rest approach and an optimize binary SVM,
this is a good hello-world example for structured predicition and using pystruct.
In the case of multi-class SVMs, in contrast to more structured models, the
labels set Y is just the number of classes, so inference can be performed by
just enumerating Y.


Lets say we want to classify the classical iris dataset. There are three classes and four features:


The Crammer-Singer model implemented in ... is implemented as the joint feature function:


and inference is just the argmax over the three responses (one for each class):

To perform max-margin learning, we also need the loss-augmented inference. PyStruct has an optimized version,
but a pure python version would look like this:


Essentialy the response (score / energy) of wrong label is down weighted by 1, the loss of doing an incorrect prediction.
We could also implement a custom loss function, that assigns custom losses for predicting, say ... as ...

For training, we pick the n-slack SSVM, which works well with few samples and requires little tuning:


Multi-label SVM
===============
A multi-label classification task is one where each sample can be labeled with any number of classes.
In other words, there are n_classes many binary labels, each indicating whether a sample belongs
to a given class or not. This could be treated as n_classes many independed binary classification
problems, as the scikit-learn OneVsRest classifier does.
However, it might be beneficial to exploit correlations between labels to achieve better generalization.

In the scene classification dataset, each sample is a picture of an outdoor scene,
representated using simple color aggregation. The labels characterize the kind of scene, which can be
"beach", "sunset", "fall foilage", "field", "mountain" or "urban". Each image can belong to multiple classes,
such as "fall foilage" and "field". Clearly some combinations are more likely than others.

We could try to model all possible combinations, which would result in a 2 ** 6
= 64 class multi-class classification problem. This would allow us explicitly model all correlations between labels,
but it would prevent us from predicting combinations that don't appear in the training set.
Even if a combination did appear in the training set, the numer of samples in each class would be very small.
A compromise between modeling all correlations and modelling no correlations is modeling only pairwise correlations,
which is the approach implemented in :class:`models.MultiLabelClf`.
It creates a graph over ``n_classes`` binary nodes, together with edges between each pair of classes.
Each binary node has represents one class, and therefor will get its own column
in the weight-vector, similar to the crammer-singer multi-class classification.

In addition, there is a pairwise weight betweent each pair of labels.
This leads to a feature function of this form:

If our graph has only 6 nodes, we can actually enumerate all states.
Unfortunately, in general, inference in a fully connected binary graph is in
gerneral NP-hard, so we might need to rely on approximate inference, like loopy believe propagation or AD3.
#FIXME do enumeration! benchmark!!

An alternative to using approximate inference for larger numbers of labels is to not create a fully connected graph,
but restrict ourself to pairwise interactions on a tree over the labels. In the above example of outdoor scenes,
some labels might be informative about others, maybe a beach picture is likely to be of a sunset, while
an urban scene might have as many sunset as non-sunset samples. The optimum tree-structure for such a problem
can easily be found using the Chow-Liu tree, which is simply the maximum weight spanning tree over the graph, where
edge-weights are given by the mutual information between labels on the training set.
You can use the Chow-Liu tree method simply by specifying ``edges="chow_liu"``.
This allows us to use efficient and exact max-product message passing for
inference.

#FIXME sample

The implementation of the inference for this model creates a graph with unary
potentials (given by the inner product of features and weights), and pairwise
potentials given by the pairwise weight. This graph is then passed to the
general graph-inference, which runs the selected algorithm.


Conditional-Random-Field-like graph models
==========================================
The following models are all pairwise models over nodes, that is they model a labeling of a graph,
using features at the nodes, and relation between neighboring nodes.
The main assumption in these models in PyStruct is that nodes are homogeneous, that is they all
have the same meaning. That means that each node has the same number of classes, and these classes
mean the same thing. In practice that means that weights are shared across all nodes and edges,
and the model adapts via features.
This is in contrast to the :class:`MultiLabelClf`, which builds a binary graph
were nodes mean different things (each node represents a different class), so they do not share weights.

#FIXME alert!
I call these models Conditional Random Fields (CRFs), but this a slight abuse of notation,
as PyStruct actually implements perceptron and max-margin learning, not maximum likelihood learning.
So these models might better be called Maximum Margin Random Fields. However, in the computer vision
community, it seems most pairwise models are called CRFs, independent of the method of training.

Chain CRF
----------
One of the most common use-cases for structured prediction is chain-structured
outputs. These occur naturaly in sequence labeling tasks, such as
Part-of-Speech tagging or named entity recognition in natural language
processing, or segmentation and phoneme recognition in speech processing.

FIXME alert
While pystruct is able to work with chain CRFs, it is not explicitly built with these in mind,
and there are libraries that optimize much more for this special case, such as seqlearn and CRF++.

models
-------
how to use

multi-class
multi-label
chain-crf
graph-crf
edge-feature graph crf


how to write your own model
----------------------------


tips on solvers

tips on inference algorithms
