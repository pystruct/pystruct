"""
==============================================
Conditional Interactions on the Snakes Dataset
==============================================
This example uses the snake dataset introduced in
Nowozin, Rother, Bagon, Sharp, Yao, Kohli: Decision Tree Fields ICCV 2011

This dataset is specifically designed to require the pairwise interaction terms
to be conditioned on the input, in other words to use non-trival edge-features.

The task is as following: a "snake" of length ten wandered over a grid. For
each cell, it had the option to go up, down, left or right (unless it came from
there). The input consists of these decisions, while the desired output is an
annotation of the snake from 0 (head) to 9 (tail).  See the plots for an
example.

As input features we use a 3x3 window around each pixel (and pad with background
where necessary). We code the five different input colors (for up, down, left, right,
background) using a one-hot encoding. This is a rather naive approach, not using any
information about the dataset (other than that it is a 2d grid).

The task can not be solved using the simple DirectionalGridCRF - which can only
infer head and tail (which are also possible to infer just from the unary
features). If we add edge-features that contain the features of the nodes that are
connected by the edge, the CRF can solve the task.

From an inference point of view, this task is very hard.  QPBO move-making is
not able to solve it alone, so we use the relaxed AD3 inference for learning.

PS: This example runs a bit (5 minutes on 12 cores, 20 minutes on one core for me).
But it does work as well as Decision Tree Fields ;)

UPDATE: we also inject domain knowledge at inference time by telling that there 
is at-most or exactly one of each annotation from 1 to 10  (0 is background).

    JL Meunier - January 2017
    
    Developed for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943
    
    Copyright Xerox

"""
import time
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score

from pystruct.learners import OneSlackSSVM
from pystruct.datasets import load_snakes
from pystruct.models import EdgeFeatureGraphCRF

from plot_snakes import one_hot_colors, prepare_data

def REPORT(l_Y_GT, lY_Pred, t=None):
    if t: print "\t( predict DONE IN %.1fs)"%t
        
    _flat_GT, _flat_P = (np.hstack([y.ravel() for y in l_Y_GT]),  
                         np.hstack([y.ravel() for y in lY_Pred]))
    confmat = confusion_matrix(_flat_GT, _flat_P)
    print confmat
    print "\ttrace   =", confmat.trace()
    print "\tAccuracy= %.3f"%accuracy_score(_flat_GT, _flat_P)    


print("Please be patient. Learning will take 5-20 minutes.")
snakes = load_snakes()
X_train, Y_train = snakes['X_train'], snakes['Y_train']
#X_train, Y_train = X_train[:5], Y_train[:5]

X_train = [one_hot_colors(x) for x in X_train]
Y_train_flat = [y_.ravel() for y_ in Y_train]

X_train_directions, X_train_edge_features = prepare_data(X_train)
print "%d picture for training"%len(X_train)

# Evaluate using confusion matrix.
# Clearly the middel of the snake is the hardest part.
X_test, Y_test = snakes['X_test'], snakes['Y_test']
X_test = [one_hot_colors(x) for x in X_test]
Y_test_flat = [y_.ravel() for y_ in Y_test]
X_test_directions, X_test_edge_features = prepare_data(X_test)
print "%d picture for test"%len(X_test)


print "- TRAINING ONLY WITH DIRECTIONAL EDGE FEATURES -----"
#inference = 'qpbo'
#I'm interested in AD3 inference.
inference = 'ad3'

# first, train on X with directions only:
crf = EdgeFeatureGraphCRF(inference_method=inference)
ssvm = OneSlackSSVM(crf, inference_cache=50, C=.1, tol=.1, max_iter=100,
                    n_jobs=1)
t0 = time.time()
ssvm.fit(X_train_directions, Y_train_flat)
print("Model EdgeFeatureGraphCRF fitted. %.1fs"%(time.time()-t0))

Y_GT = np.hstack(Y_test_flat)
print("- Results using only directional features for edges. %.1fs"%(time.time()-t0))
t0 = time.time()
Y_pred = ssvm.predict(X_test_directions)
REPORT(Y_GT, Y_pred, time.time()-t0)

print "- Result with binarized graph"
t0 = time.time()
Y_pred = ssvm.predict(X_test_directions, [True]*len(X_test_directions))
REPORT(Y_GT, Y_pred, time.time()-t0)


#Predict under constraints
def buildConstraints(X, bOne=True):
    """
    We iterate over each graph, and make sure that for each, we constrain to have a single instances of classes 1 to 9
    (or atmost one) 
    
    The constraints must be a list of tuples like ( <operator>, <unaries>, <states>, <negated> )
    where:
    - operator is one of 'XOR' 'XOROUT' 'ATMOSTONE' 'OR' 'OROUT' 'ANDOUT' 'IMPLY'
    - unaries is a list of the index of the unaries involved in this constraint
    - states is a list of unary states, 1 per involved unary. If the states are all the same, you can pass it directly as a scalar value.
    - negated is a list of boolean indicated if the unary must be negated. Again, if all values are the same, pass a single boolean value instead of a list 
    """
    sLogicOp = "XOR" if bOne else "ATMOSTONE"
    lConstraint = []
    for (node_features, edges, edge_features) in X:
        n_nodes = node_features.shape[0]
        lConstraintPerGraph = [ (sLogicOp, range(n_nodes), i, False) for i in range(1,10) ] #only one 
        lConstraint.append( lConstraintPerGraph )
    return lConstraint
        

print "- Results of inference under constraints"
lConstraint = buildConstraints(X_test_directions)
t0 = time.time()
Y_pred = ssvm.predict(X_test_directions, lConstraint)
REPORT(Y_GT, Y_pred, time.time()-t0)

# now, use more informative edge features:
print "- NOW TRAINING WITH BETTER EDGE FEATURES -----"
inference = 'qpbo'
crf = EdgeFeatureGraphCRF(inference_method=inference)
ssvm = OneSlackSSVM(crf, inference_cache=50, C=.1, tol=.1, switch_to='ad3',
                    n_jobs=1)
t0 = time.time()
ssvm.fit(X_train_edge_features, Y_train_flat)
print("Model EdgeFeatureGraphCRF fitted. %.1fs"%(time.time()-t0))


print("- Results using also input features for edges. %.1fs"%(time.time()-t0))
t0 = time.time()
Y_pred = ssvm.predict(X_test_edge_features)
REPORT(Y_GT, Y_pred, time.time()-t0)

print "- Result with binarized graph"
t0 = time.time()
Y_pred = ssvm.predict(X_test_edge_features, [True]*len(X_test_edge_features))
REPORT(Y_GT, Y_pred, time.time()-t0)

#Predict under constraints
print "- Results of inference under constraints"
lConstraint = buildConstraints(X_test_edge_features)
t0 = time.time()
Y_pred = ssvm.predict(X_test_edge_features, lConstraint)
REPORT(Y_GT, Y_pred, time.time()-t0)


"""
Please be patient. Learning will take 5-20 minutes.
200 picture for training
100 picture for test
- TRAINING ONLY WITH DIRECTIONAL EDGE FEATURES -----
Model EdgeFeatureGraphCRF fitted. 115.9s
- Results using only directional features for edges. 115.9s
    ( predict DONE IN 0.7s)
[[2750    0    0    0    0    0    0    0    0    0    0]
 [   0  100    0    0    0    0    0    0    0    0    0]
 [   0    0   59    0   22    4    6    1    7    1    0]
 [   0    3    1   29    5   31    8   18    1    3    1]
 [   0    1   13    2   30   11   25    1   13    3    1]
 [   0    1    1    9    4   46   11   15    3    9    1]
 [   0    1    7    2   24   10   21    7   23    2    3]
 [   0    0    1    6    7   35   10   17    3   21    0]
 [   0    0    7    2   14   10   16    4   25    0   22]
 [   0    0    0    3    7   14    4   12    2   58    0]
 [   0    0    5    3   11    3    7    0    5    0   66]]
    trace   = 3201
    Accuracy= 0.854
- Result with binarized graph
    ( predict DONE IN 0.7s)
[[2750    0    0    0    0    0    0    0    0    0    0]
 [   0  100    0    0    0    0    0    0    0    0    0]
 [   0    0   59    0   22    4    6    1    7    1    0]
 [   0    3    1   29    5   31    8   18    1    3    1]
 [   0    1   13    2   30   11   25    1   13    3    1]
 [   0    1    1    9    4   46   11   15    3    9    1]
 [   0    1    7    2   24   10   21    7   23    2    3]
 [   0    0    1    6    7   35   10   17    3   21    0]
 [   0    0    7    2   14   10   16    4   25    0   22]
 [   0    0    0    3    7   14    4   12    2   58    0]
 [   0    0    5    3   11    3    7    0    5    0   66]]
    trace   = 3201
    Accuracy= 0.854
- Results of inference under constraints
    ( predict DONE IN 0.7s)
[[2750    0    0    0    0    0    0    0    0    0    0]
 [   0  100    0    0    0    0    0    0    0    0    0]
 [   0    0   59    0   22    4    6    1    7    1    0]
 [   0    3    1   29    5   31    8   18    1    3    1]
 [   0    1   13    2   30   11   25    1   13    3    1]
 [   0    1    1    9    4   46   11   15    3    9    1]
 [   0    1    7    2   24   10   21    7   23    2    3]
 [   0    0    1    6    7   35   10   17    3   21    0]
 [   0    0    7    2   14   10   16    4   25    0   22]
 [   0    0    0    3    7   14    4   12    2   58    0]
 [   0    0    5    3   11    3    7    0    5    0   66]]
    trace   = 3201
    Accuracy= 0.854
- NOW TRAINING WITH BETTER EDGE FEATURES -----
Model EdgeFeatureGraphCRF fitted. 679.6s
- Results using also input features for edges. 679.6s
    ( predict DONE IN 0.9s)
[[2749    0    0    0    0    0    0    0    1    0    0]
 [   0  100    0    0    0    0    0    0    0    0    0]
 [   0    0  100    0    0    0    0    0    0    0    0]
 [   0    0    0  100    0    0    0    0    0    0    0]
 [   0    0    0    0   98    0    1    0    1    0    0]
 [   0    0    0    2    0   98    0    0    0    0    0]
 [   0    0    0    0    2    0   98    0    0    0    0]
 [   0    1    0    0    0    2    0   97    0    0    0]
 [   0    0    1    0    0    0    1    0   98    0    0]
 [   0    0    0    1    0    0    0    0    0   99    0]
 [   0    0    0    0    1    0    0    0    0    0   99]]
    trace   = 3736
    Accuracy= 0.996
- Result with binarized graph
    ( predict DONE IN 0.9s)
[[2749    0    0    0    0    0    0    0    1    0    0]
 [   0  100    0    0    0    0    0    0    0    0    0]
 [   0    0  100    0    0    0    0    0    0    0    0]
 [   0    0    0  100    0    0    0    0    0    0    0]
 [   0    0    0    0   98    0    1    0    1    0    0]
 [   0    0    0    2    0   98    0    0    0    0    0]
 [   0    0    0    0    2    0   98    0    0    0    0]
 [   0    1    0    0    0    2    0   97    0    0    0]
 [   0    0    1    0    0    0    1    0   98    0    0]
 [   0    0    0    1    0    0    0    0    0   99    0]
 [   0    0    0    0    1    0    0    0    0    0   99]]
    trace   = 3736
    Accuracy= 0.996
- Results of inference under constraints
    ( predict DONE IN 0.9s)
[[2749    0    0    0    0    0    0    0    1    0    0]
 [   0  100    0    0    0    0    0    0    0    0    0]
 [   0    0  100    0    0    0    0    0    0    0    0]
 [   0    0    0  100    0    0    0    0    0    0    0]
 [   0    0    0    0   98    0    1    0    1    0    0]
 [   0    0    0    2    0   98    0    0    0    0    0]
 [   0    0    0    0    2    0   98    0    0    0    0]
 [   0    1    0    0    0    2    0   97    0    0    0]
 [   0    0    1    0    0    0    1    0   98    0    0]
 [   0    0    0    1    0    0    0    0    0   99    0]
 [   0    0    0    0    1    0    0    0    0    0   99]]
    trace   = 3736
    Accuracy= 0.996


"""