
[![Build Status](https://travis-ci.org/jlmeunier/pystruct.png)](https://travis-ci.org/jlmeunier/pystruct)
[![pypi version](http://img.shields.io/pypi/v/pystruct.svg?style=flat)](https://pypi.python.org/pypi/pystruct/)
[![licence](http://img.shields.io/badge/licence-BSD-blue.svg?style=flat)](https://github.com/pystruct/pystruct/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/21369/pystruct/pystruct.svg)](https://zenodo.org/badge/latestdoi/21369/pystruct/pystruct)



# PyStruct+
This is a fork from Andreas Mueller's [pystruct](https://github.com/pystruct/pystruct) project, which is an easy-to-use structured learning
 and prediction library. In particular, pystruct provides a well-documented tool for researchers as well as non-experts to make use of structured 
 prediction algorithms. And the design tries to stay as close as possible to the interface and conventions of [scikit-learn](http://scikit-learn.org).

The goal of the [pystruct+](https://github.com/jlmeunier/pystruct) project is to extend pystruct along two directions:
 * **supporting hard-logic constraints when predicting**
 * **supporting nodes of different nature in CRF graphs**
 
 By-products of this fork are:
 * Python 3 compatibility
 * Unit tests passing again
 
The extension is 100% ascendant compatible with pystruct. Anything that you did with pystruct works the same way with pystruct+. 
So you can refer to the pystruct documentation for the API, examples, etc. ( http://pystruct.github.io )

What is different in pystruct+?
 * the __*predict*__ method accepts now an optional constraint parameter
 * a new CRF model is proposed, __*NodeTypeEdgeFeatureGraphCRF*__

 More details are given in next sections.
 
 You can contact the author on [github](https://github.com/jlmeunier/pystruct). Comments and contributions are welcome.
 
# Credit to EU READ Project
Developed  for the EU project READ. The READ project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 674943.
 
 
## Installation
This extension requires **ad3** version 2.2 (so for now, Feb 15th, 2018, you need to get it directly from https://github.com/andre-martins/AD3 )

The support of hard-logic constraint requires you to choose as solver "ad3+". This is still ad3 code, but working on a binarized graph.

For learning I mostly used the __*OneSlackSSVM*__ learner, which requires to install **cvxopt** as well.

### Python libraries:
> pip install install numpy scipy cvxopt pyqpbo scikit-learn nose pytest 

### For AD3:
 * get it from https://github.com/andre-martins/AD3
 * install it: 

> python setup.py install

### For Pystruct+:
 * get the source code from https://github.com/jlmeunier/pystruct
 * compile and install:

> python setup.py install

## Tests
To test your install, run the test of the new CRF model:
> python pystruct/tests/test_models/test_node_type_edge_feature_graph_crf.py

(You should see a "OK" displayed at the end of the script execution.)

## Example
Building on the [Snakes](https://pystruct.github.io/auto_examples/plot_snakes.html#sphx-glr-auto-examples-plot-snakes-py) example, there is now a new example called "HiddenSnakes". (Code in examples/plot_hidden_short_snakes_typed.py )

The idea is that some picture do not contain any snake despite 10 pixels have a Snake body colour. Why? Because they do not form a valid 10-long snake, as 1 pixel has a wrong colour destroying the continuity of the snake.

The original task remains but is more difficult: some non-blue pixels are now labelled 'background'. An additional task consists in labeling the picture as Snake or NoSnake.

This double task is solved by the use of an additional type of node that represents the picture itself, with 7 simplistic features. There are additional edges, from each pixel to the picture node. That's all. And it improves a lot from the results of the *EdgeFeatureGraphCRF*-based model. 

In addition, we injected some more domain knowledge to illustrate the use of the hard logic constraints. In this case we enforce *at most one pixel of label L per picture, for L in [1, 10]*. This gives an extra accuracy bonus.

## Prediction with Hard-Logic Constraints

You can now pass a __list of logical constraints__ to the predict method, with a *constraints=* named parameter.

    Each constraint is tuple like *( operator, nodes, labels, negated )*
    where:
    - *operator* is one of 'XOR' 'XOROUT' 'ATMOSTONE' 'OR' 'OROUT' 'ANDOUT' 'IMPLY'
    - *nodes* is the list of the index of each node involved in this constraint
    - *labels* is the list of node label. If the labels are the same for all nodes, you can pass it directly as a scalar value.
    - *negated* is a list of boolean indicating if the corresponding argument must be negated. Again, if all values are the same, pass a single boolean value instead of a list.

The operators whose name ends with 'OUT' impose that the operator applied on the all-but-last arguments yields the truth value of the last one.
>For instance XOROUT(a,b,c) <=> XOR(a,b) = c

When used jointly with the new *NodeTypeEdgeFeatureGraphCRF* model, the structure of the constraints list slightly differs. See in next section.


## CRF Graph with Nodes of Different Nature
Pystruct CRF graphs assumes that the nodes of the graph all have the same nature. In consequence, all nodes share the same weights and the same set of possible labels. Similarly, all edges have the same nature and share the same edge weights.
This was a limitation with regards to our needs (for a Document Understanding task). So we propose a new CRF model called *NodeTypeEdgeFeatureGraphCRF*.

*NodeTypeEdgeFeatureGraphCRF* supports multiple node of multiple nature, which we call **node types**. Each type has its own weights and own set of possible labels. Similarly, edges have different nature depending on the type of their sources and target-nodes. In a graph with N types, there are N^2 types of edges.

*NodeTypeEdgeFeatureGraphCRF* generalizes *EdgeFeatureGraphCRF*, so edges have features. NOTE: I think that you can mimics the absence opf feature on edges (as in *GraphCRF* model) by specifying one feature per edge, whose value is 1 for all edges.

**This extension has an impact on:**
 * the constructor
 * the structure of the label weights, if not uniform
 * the structure of the Xs
 * the values in Ys
 * the structure of the optional constraint list at prediction

### Class Constructor
You need now to define the number of node types and the number of features per type (of node, and of edge) when instantiating *NodeTypeEdgeFeatureGraphCRF*.

    def __init__(self
                 , n_types                  #how many node type?
                 , l_n_states               #how many labels   per node type?
                 , l_n_features             #how many features per node type?
                 , a_n_edge_features        #how many features per edge type? (array-like) shape=(n_type, n_type) -> n_feature_per_type_pair
                 , inference_method="ad3" 
                 , l_class_weight=None):    #class_weight      per node type or None     <list of array-like> or None
 

### Xs and Ys
In single type CRF, like *EdgeFeatureGraphCRF*, an instance *X* is represented as a tuple        

    (*node_features*, *edges*, *edge_features*) representing the graph. 

* *node_feature*s is of shape (*n_node*, *n_features*)
* *edges* is an array of shape (*n_edges*, 2)
* *edge_features* is of shape (*n_edges*, *n_edge_features*)

    Labels y are given as array of shape (*n_nodes*,)

In multiple type graphs, with *_n_types* types, an instance *X* is represented as a tuple

     (*l_node_features*, *l_edges*, *l_edge_features*) representing the graph. 
* *l_node_feature*s is a list of length *n_types* containing arrays of shape (*n_typ_node*, *n_typ_features*), where *n_typ_node* is the number of nodes of that type, while *n_typ_features* is the number of features for this type of nodes.
* *l_edges* is a list of length *n_types*^2 . Each of its elements contains an array of shape (*n_typ_edge, 2) defining the edges from nodes of type *i* to nodes of type *j*, with *i* and *j* in [0, *n_types*-1], *j* being the secondary index (inner loop). The index of the nodes in each type starts at 0.
* *l_edge_features* is a list of length *n_types*^2. It contains the features of the edges for each pair of types, in same order as in previous parameter. Each item is an array of shape (*n_typ_edges*, *n_typ_edge_features*). if *n_typ_edge_features* is 0, then *n_typ_edges* should be 0 as well for all instances of graph! If you want an edge without features, set *n_typ_edge_features* to 1 and pass 1.0 as feature for all edges (of that type).

Each *Y* remains a vector array. While the label could start at 0 for all types, we have chosen not to do so. (Essentially,to make clear that types do not blend into each other, which is clear when you show a confusion matrix). So the labels of the first type start at 0, while labels of next type starts right after the value of the last label of previous type.
*NodeTypeEdgeFeatureGraphCRF* provides 2 convenience methods:
* *flattenY*( [ [2,0,0], [3,3,4] ] ) --> [ 2,0,0, 5,5,7]  (assuming type 0 has 3 labels)
* *unflattenY*(Xs, [ 2,0,0, 5,5,7] ) --> [ [2,0,0], [3,3,4] ]   (you'll also need to pass the Xs)

### Constraints on Multitype Graphs
As for the Xs and Ys, the constraint must be partitioned by type.

    The constraints must be a list of tuples like:
  
Either

    ( *operator*, *l_nodes*, *l_labels*, *l_negated* )
     with operator being one 'XOR' 'ATMOSTONE' 'OR'

Or

    ( *operator*, *l_nodes*, *l_labels*, *l_negated* , (*type*, *node*, *label*, *negated*))
     with operator being one  'XOROUT' 'OROUT' 'ANDOUT' 'IMPLY'
    
- *l_nodes* is a list of nodes per type. Each item is a list of the index of the node of that type involved in this constraint
- *l_labels* is a list of labels per type. Each item is a list of the label of the involved node. If the labels are all the same for a type, you can pass it directly as a scalar value.
- *l_negate*d is a list of "negated" per type. Each item is a list of booleans indicating if the node must be negated. Again, if all values are the same for a type, pass a single boolean value instead of a list 

- the last (*type*, *nod*e, *label*, *negated*) allows to refer to the outcome of an 'OUT' operator.
   
   
