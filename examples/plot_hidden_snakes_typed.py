"""
==============================================
Conditional Interactions on the Snakes Dataset
==============================================

This is a variant of plot_snakes.py 

Snake are hidding, so another task is both to determine if a snake is in the picture, and
identify its head to tail body.  

We use the NodeTypeEdgeFeatureGraphCRF class with 2 type of nodes.


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
import random
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, accuracy_score
import time
import sys

from pystruct.learners import OneSlackSSVM
from pystruct.datasets import load_snakes
from pystruct.utils import make_grid_edges, edge_list_to_features
#from pystruct.models import EdgeFeatureGraphCRF
from pystruct.models import NodeTypeEdgeFeatureGraphCRF

from plot_snakes import one_hot_colors, neighborhood_feature, prepare_data

from plot_hidden_snakes import shufflePictureCells, shuffleSnakeCells, changeOneSnakeCell, augmentWithNoSnakeImages, shuffle_in_unison

def shuffleSnake(a_picture, bOneHot=True):
    """
    Shuffle either the snake's cells or the pcitures' pixels.
    """
    if True:
        changeOneSnakeCell(a_picture, bOneHot)
        changeOneSnakeCell(a_picture, bOneHot)
    else:
        if random.randint(0,1):
            shuffleSnakeCells(a_picture, bOneHot)
        else:
            shufflePictureCells(a_picture)
    
def plot_snake(picture):
    plt.imshow(picture, interpolation='nearest')
    plt.show()

def prepare_picture_data(X):
    """
    compute picture features (on 1-hot encoded pictures)
    """
    lPictFeat = list()
    for a_hot_picture in X:
        #count number of cells of each color
        #feat = np.zeros((1,5), dtype=np.int8)
        feat = np.zeros((1,7), dtype=np.int64)

        #Histogram of pixels from 0 to 4
        """
        Test accuracy: 0.500
        [[45 55]
         [45 55]]
        """
        for i in xrange(5):
            ai, aj = np.where(a_hot_picture[...,i] == 1)
            feat[0,i] = len(ai)

        #adding height and width of the snake
        """
        Test accuracy: 0.420    Test accuracy: 0.515    Test accuracy: 0.495
        [[39 61]                [[48 52]                [[52 48]
         [55 45]]                [45 55]]                [53 47]]
        """
        ai, aj = np.where(a_hot_picture[...,3] != 1)
        feat[0,5] = max(ai)-min(ai)     #height
        feat[0,6] = max(aj)-min(aj)     #width
        
        lPictFeat.append(feat)
    
    return lPictFeat

def convertToTwoType(X_train,               #list of hot pictures
                     X_train_directions,    # list of node_feat (2D array) , edges (_ x 2 array), edge_feat (2D array) for pixel nodes
                     Y_train,               # list of 2D arrays    
                     X_train_pict_feat,         #a list of picture_node_features
                     Y_train_pict):             #a list of integers [0,1]
    """
    return X,Y for NodeTypeEdgeFeatureGraphCRF
    
    
    X and Y
    -------
    Node features are given as a list of n_types arrays of shape (n_type_nodes, n_type_features):
        - n_type_nodes is the number of nodes of that type
        - n_type_features is the number of features for this type of node
    
    Edges are given as a list of n_types x n_types arrays of shape (n_type_edges, 2). 
        Columns are resp.: node index (in corresponding node type), node index (in corresponding node type)
    
    Edge features are given as a list of n_types x n_types arrays of shape (n_type_type_edge, n_type_type_edge_features)
        - n_type_type_edge is the number of edges of type type_type
        - n_type_type_edge_features is the number of features for edge of type type_type
        
    An instance ``X`` is represented as a tuple ``([node_features, ..], [edges, ..], [edge_features, ..])`` 

    Labels ``Y`` are given as one array of shape (n_nodes)   The meaning of a label depends upon the node type. 
    
    """
    
    lX, lY = list(), list()
    
    for (X,
        (aPixelFeat, aPixelPixelEdges, aPixelPixelEdgeFeat),
        aPixelLbl,
        aPictFeat,
        iPictLbl) in zip(X_train, X_train_directions, Y_train, X_train_pict_feat, Y_train_pict ):


        aPixelPictEdges = np.zeros( (aPixelFeat.shape[0], 2), np.int64)
        aPixelPictEdges[:,0] = np.arange(aPixelFeat.shape[0])
        features = neighborhood_feature(X)
        aPixelPictEdgeFeat = features
        
        lNodeFeat   = [aPixelFeat, aPictFeat]
        lEdge       = [aPixelPixelEdges,
                       aPixelPictEdges,       #pixel to picture
                       None,                  #picture to pixel
                       None]                  #picture to picture
        lEdgeFeat   = [aPixelPixelEdgeFeat, 
                       aPixelPictEdgeFeat,
                       None,
                       None]
        
        #Y is flat for each graph
        y = np.zeros((aPixelLbl.size+1, ), dtype=np.int64)
        y[:-1] = aPixelLbl.ravel()
        y[-1]  = int(iPictLbl)+11
        
        x = (lNodeFeat, lEdge, lEdgeFeat)
        
        lX.append(x)
        lY.append(y)
        
    return lX,lY





if __name__ == '__main__':
    
    np.random.seed(1605)
    random.seed(98)
    
    print("Please be patient...")
    snakes = load_snakes()
    X_train, Y_train = snakes['X_train'], snakes['Y_train']
    
    bADD_HIDDEN_SNAKES = True
    #bADD_HIDDEN_SNAKES = False
    #JL
    X_train, Y_train = X_train[:3], Y_train[:3]
    print len(X_train), len(Y_train)
    #print `X_train[0]`

    if bADD_HIDDEN_SNAKES:
        nb_hidden, X_train, Y_train = augmentWithNoSnakeImages(X_train, Y_train, "train", False)
        print len(X_train), len(Y_train)
        Y_train_pict = np.array([1]*(len(X_train)-nb_hidden) + [0]*nb_hidden)
    
    X_train = [one_hot_colors(x) for x in X_train]

    X_train, Y_train, Y_train_pict = shuffle_in_unison(X_train, Y_train, Y_train_pict)


    X_train_pict_feat = prepare_picture_data(X_train)
 
    #X_train_pixel_pict_edge, X_train_pixel_pict_edge_feat = prepare_picture_edge_data(X_train)
    
    X_train_directions, X_train_edge_features = prepare_data(X_train)

    inference = 'ad3'
    # first, train on X with directions only:
    #crf = NodeTypeEdgeFeatureGraphCRF(1, [11], [45], [[2]], inference_method=inference)
    # first, train on X with directions only:
#     l_weights = [ 
#                     [10.0/200] + [10.0/200]*10,
#                     [10.0/20 , 10.0/20]
#             ]
#     print "WEIGHTS:", l_weights
    crf = NodeTypeEdgeFeatureGraphCRF(2,        # 2 node types: pixels and pictures
                                      [11, 2],  # 11 states for pixel nodes, 2 states for pictures
                                      [45, 7],  # 45 features for pixels, 7 for pictures
                                      [[180, 45],      # 2 feature between pixel nodes, 1 between pixel and picture
                                       [45  , 0]],   # , nothing between picture nodes (no picture_to_picture edge anyway)
                                      inference_method=inference
#                                       , l_class_weight = l_weights 
                                      )
    ssvm = OneSlackSSVM(crf, inference_cache=50, C=.1, tol=.1,  
                        #max_iter=1000,
                        n_jobs=1
                        ,verbose=1
                        )
                                                                
    print "YY[0].shape", Y_train[0].shape
    XX, YY = convertToTwoType(X_train,
                             X_train_edge_features,    # list of node_feat , edges, edge_feat for pixel nodes
                             Y_train,
                             X_train_pict_feat,         #a list of picture_node_features
                             Y_train_pict)              #a list of integers [0,1]
    
    print  np.histogram(   np.hstack([y.ravel() for y in YY]), bins=range(14))
#     print  np.histogram(   np.hstack([y.ravel()[:-1] for y in YY]), bins=range(12))
#     print  np.histogram(   np.hstack([y.ravel()[-1]  for y in YY]), bins=range(3))
#     yy_trn      = np.hstack([y.ravel()[:-1] for y in YY])
#     print(confusion_matrix(yy_trn,yy_trn))
#     yy_trn_pic  = np.hstack([y.ravel()[-1]  for y in YY])
#     print(confusion_matrix(np.hstack(yy_trn_pic), np.hstack(yy_trn_pic)))
    
    
    print "YY[0].shape", YY[0].shape
    crf.initialize(XX, YY)# check if the data is properly built
    sys.stdout.flush()
    
    t0 = time.time()
    ssvm.fit(XX, YY)
    print "FIT DONE IN %.1fs"%(time.time() - t0)
    sys.stdout.flush()
    
#     import sys
#     sys.exit(0)
    
    # Evaluate using confusion matrix.
    # Clearly the middel of the snake is the hardest part.
    X_test, Y_test = snakes['X_test'], snakes['Y_test']
#     X_test, Y_test = X_test[:3], Y_test[:3]

    if bADD_HIDDEN_SNAKES:
        nb_hidden, X_test, Y_test = augmentWithNoSnakeImages(X_test, Y_test, "test", False)
        print len(X_test), len(Y_test)
        Y_test_pict = np.array([1]*(len(X_test)-nb_hidden) + [0]*nb_hidden)
    
    X_test = [one_hot_colors(x) for x in X_test]

    #useless X_test, Y_test, Y_test_pict = shuffle_in_unison(X_test, Y_test, Y_test_pict)

    X_test_pict_feat = prepare_picture_data(X_test)
    
    X_test_directions, X_test_edge_features = prepare_data(X_test)
    
    XX_test, YY_test =convertToTwoType(X_test,
                     X_test_edge_features,    # list of node_feat , edges, edge_feat for pixel nodes
                     Y_test,
                     X_test_pict_feat,         #a list of picture_node_features
                     Y_test_pict)              #a list of integers [0,1]
    
    print  np.histogram(   np.hstack([y.ravel() for y in YY_test]), bins=range(14))

    YY_pred = ssvm.predict( XX_test )
    print len(XX_test), len(YY_pred)
    
    print confusion_matrix(np.hstack([y.ravel() for y in YY_test]),
                           np.hstack([y.ravel() for y in YY_pred]))

#     Y_test_flat = np.hstack([y.ravel()[:-1] for y in YY_test])
#     Y_pred_flat = np.hstack([y.ravel()[:-1] for y in YY_pred])
#     
#     print("Results using only relevant features for edges")
#     print("Test accuracy: %.3f"
#           % accuracy_score(Y_test_flat, Y_pred_flat))
#     print(confusion_matrix(Y_test_flat, Y_pred_flat))
# 
#     Y_pict_pred = [yy.ravel()[-1]  for yy in YY_pred]
#     print("Results AT PICTURE LEVEL using only directional features for edges")
#     print("Test accuracy: %.3f"
#           % accuracy_score(Y_test_pict, Y_pict_pred))
#     print(confusion_matrix(Y_test_pict, Y_pict_pred))
       
    
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

    print "DONE"


"""

 


"""