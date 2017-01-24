"""
==============================================
Conditional Interactions on the Snakes Dataset
==============================================

This is a variant of plot_snakes.py where we use the NodeTypeEdgeFeatureGraphCRF
class instead of EdgeFeatureGraphCRF, despite there is only 1 type of nodes.
So this should give exact same results as plot_snakes.py


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

    JL Meunier - January 2017
    
    Developed for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943
    
    Copyright Xerox

"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, accuracy_score

from pystruct.learners import OneSlackSSVM
from pystruct.datasets import load_snakes
from pystruct.utils import make_grid_edges, edge_list_to_features
#from pystruct.models import EdgeFeatureGraphCRF
from pystruct.models import NodeTypeEdgeFeatureGraphCRF

from plot_snakes import  one_hot_colors, neighborhood_feature, prepare_data

def convertToSingleTypeX(X):
    """
    For NodeTypeEdgeFeatureGraphCRF X is structured differently.
    But NodeTypeEdgeFeatureGraphCRF can handle graph with a single node type. One needs to convert X to the new structure using this method.
    """
    return [([nf], [e], [ef]) for (nf,e,ef) in X]

if __name__ == '__main__':
    print("Please be patient. Learning will take 5-20 minutes.")
    snakes = load_snakes()
    X_train, Y_train = snakes['X_train'], snakes['Y_train']
    
    X_train = [one_hot_colors(x) for x in X_train]
    Y_train_flat = [y_.ravel() for y_ in Y_train]


    X_train_directions, X_train_edge_features = prepare_data(X_train)

    inference = 'qpbo'
    # first, train on X with directions only:
    crf = NodeTypeEdgeFeatureGraphCRF(1, [11], [45], [[2]], inference_method=inference)
    ssvm = OneSlackSSVM(crf, inference_cache=50, C=.1, tol=.1,  max_iter=100,
                        n_jobs=1)
    ssvm.fit(convertToSingleTypeX(X_train_directions), Y_train_flat)
    
    # Evaluate using confusion matrix.
    # Clearly the middel of the snake is the hardest part.
    X_test, Y_test = snakes['X_test'], snakes['Y_test']
    X_test = [one_hot_colors(x) for x in X_test]
    Y_test_flat = [y_.ravel() for y_ in Y_test]
    X_test_directions, X_test_edge_features = prepare_data(X_test)
    Y_pred = ssvm.predict( convertToSingleTypeX(X_test_directions) )
    print("Results using only directional features for edges")
    print("Test accuracy: %.3f"
          % accuracy_score(np.hstack(Y_test_flat), np.hstack(Y_pred)))
    print(confusion_matrix(np.hstack(Y_test_flat), np.hstack(Y_pred)))
       
    # now, use more informative edge features:
    crf = NodeTypeEdgeFeatureGraphCRF(1, [11], [45], [[180]], inference_method=inference)
    ssvm = OneSlackSSVM(crf, inference_cache=50, C=.1, tol=.1,  switch_to='ad3',
                        verbose=1,
                        n_jobs=8)
    ssvm.fit( convertToSingleTypeX(X_train_edge_features), Y_train_flat)
    Y_pred2 = ssvm.predict( convertToSingleTypeX(X_test_edge_features) )
    print("Results using also input features for edges")
    print("Test accuracy: %.3f"
          % accuracy_score(np.hstack(Y_test_flat), np.hstack(Y_pred2)))
    print(confusion_matrix(np.hstack(Y_test_flat), np.hstack(Y_pred2)))
    
    if False:
        # plot stuff
        fig, axes = plt.subplots(2, 2)
        axes[0, 0].imshow(snakes['X_test'][0], interpolation='nearest')
        axes[0, 0].set_title('Input')
        y = Y_test[0].astype(np.int)
        bg = 2 * (y != 0)  # enhance contrast
        axes[0, 1].matshow(y + bg, cmap=plt.cm.Greys)
        axes[0, 1].set_title("Ground Truth")
        axes[1, 0].matshow(Y_pred[0].reshape(y.shape) + bg, cmap=plt.cm.Greys)
        axes[1, 0].set_title("Prediction w/o edge features")
        axes[1, 1].matshow(Y_pred2[0].reshape(y.shape) + bg, cmap=plt.cm.Greys)
        axes[1, 1].set_title("Prediction with edge features")
        for a in axes.ravel():
            a.set_xticks(())
            a.set_yticks(())
        plt.show()

"""
Please be patient. Learning will take 5-20 minutes.
Results using only directional features for edges
Test accuracy: 0.847
[[2750    0    0    0    0    0    0    0    0    0    0]
 [   0   99    0    0    1    0    0    0    0    0    0]
 [   0    2   68    3    9    4    6    4    3    1    0]
 [   0    4   11   45    8   14    5    6    0    6    1]
 [   0    1   22   18   31    2   14    4    3    5    0]
 [   0    3    7   38   12   22    5    4    2    7    0]
 [   0    2   19   16   26    8   16    2    9    2    0]
 [   0    6   14   26   10   15    5   12    2   10    0]
 [   0    0   12   15   16    4   16    2   18    4   13]
 [   0    2    5   18    6    8    5    3    2   50    1]
 [   0    1   11    4   13    1    2    0    2    2   64]]
Results using also input features for edges
Test accuracy: 0.998
[[2749    0    0    0    0    0    0    0    1    0    0]
 [   0  100    0    0    0    0    0    0    0    0    0]
 [   0    0  100    0    0    0    0    0    0    0    0]
 [   0    0    0   99    0    0    0    0    0    1    0]
 [   0    0    0    0   99    0    1    0    0    0    0]
 [   0    0    0    1    0   98    0    1    0    0    0]
 [   0    0    0    0    1    0   99    0    0    0    0]
 [   0    0    0    0    0    1    0   99    0    0    0]
 [   0    0    0    0    0    0    0    0  100    0    0]
 [   0    0    0    0    0    0    0    1    0   99    0]
 [   0    0    0    0    0    0    0    0    0    0  100]]

"""