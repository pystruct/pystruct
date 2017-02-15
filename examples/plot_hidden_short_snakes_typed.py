"""
==============================================
Conditional Interactions on the Snakes Dataset
==============================================

This is a variant of plot_snakes.py 

Snake are hidding, so we have 2 tasks:
- determining if a snake is in the picture, 
- identifying its head to tail body.  

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

import sys, os, time
import random, cPickle

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

from pystruct.learners import OneSlackSSVM
from pystruct.datasets import load_snakes
from pystruct.models import NodeTypeEdgeFeatureGraphCRF

from plot_snakes import one_hot_colors, neighborhood_feature, prepare_data

from plot_hidden_snakes import augmentWithNoSnakeImages, shuffle_in_unison, shorten_snakes



#==============================================================================================

bFIXED_RANDOM_SEED = True  
  
NCELL=10

nbSWAP_Pixel_Pict_TYPES = 0         #0,1,2 are useful (this was for DEBUG)

bMAKE_PICT_EASY = False     #DEBUG: we had a feature on the picture that tells directly if a snake is present or not

#INFERENCE="ad3+"
INFERENCE="ad3"
N_JOBS=8

MAXITER=750

sMODELFILE = None
sMODELFILE = "model.pkl"        #we save the model in a file and do not re-trian if the file exists

#==============================================================================================

def printConfig():
    print "== NCELL=", NCELL
    print "== FIXED_SEED=", bFIXED_RANDOM_SEED
    print "== INFERENCE =", INFERENCE
    print "== N_JOBS =", N_JOBS
    print "== SWAP=", nbSWAP_Pixel_Pict_TYPES
    print "== EASY=", bMAKE_PICT_EASY
    print "== MAX_ITER=", MAXITER
    print "== MODEL FILE=", sMODELFILE

if __name__ == '__main__': printConfig()


if bFIXED_RANDOM_SEED:
    np.random.seed(1605)
    random.seed(98)
else:
    np.random.seed()
    random.seed()
    
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
                     Y_train_pict,             #a list of integers [0,1]
                     nCell=10):
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
        y[-1]  = int(iPictLbl)+nCell+1
        
        x = (lNodeFeat, lEdge, lEdgeFeat)
        
        lX.append(x)
        lY.append(y)
        
    return lX,lY

def swap_node_types(l_perm, l_n_state, lX, lY, constraints=None):
    """
    lX and lY have been produced for a CRF configured with  l_n_state
    
    We permute this as indicated by the permutation (typically for the snake: l_perm=[1, 0] ) 
    
    """
    _lX, _lY = [], []
    _constraints = None
    
    n_types = len(l_n_state)  
    a_perm = np.asarray(l_perm)                                                                       #e.g. 3 for l_n_state = [2, 3, 4]
    a_cumsum_n_state = np.asarray([sum(l_n_state[:i]) for i in range(len(l_n_state))])                           # [0, 2, 5]
    a_delta_y_by_y   = np.asarray([item for i,n in enumerate(l_n_state) for item in n*(a_cumsum_n_state[i:i+1]).tolist()])  # [0, 0, 2, 2, 2, 5, 5, 5, 5]
    a_typ_by_y       = np.asarray([item for i,n in enumerate(l_n_state) for item in n*[i]])                      # [0, 0, 1, 1, 1, 2, 2, 2, 2]
    
    _l_n_state = [l_n_state[i] for i in l_perm]
    _a_cumsum_n_state = np.asarray([sum(_l_n_state[:i]) for i in range(len(_l_n_state))])                          

    for (lNF, lE, lEF), Y in zip(lX, lY):
        
        _lNF = [lNF[i] for i in l_perm]

        _Y = np.zeros(Y.shape, dtype=Y.dtype)
        #we need to re-arrange the Ys accordingly
        l_n_nodes  = [nf.shape[0] for nf in  lNF]
        _l_n_nodes = [nf.shape[0] for nf in _lNF]
        cumsum_n_nodes  = [0] + [sum( l_n_nodes[:i+1]) for i in range(len( l_n_nodes))]
        _cumsum_n_nodes = [0] + [sum(_l_n_nodes[:i+1]) for i in range(len(_l_n_nodes))]
        for i in range(len(lNF)):
            j = l_perm[i]
            _Y[_cumsum_n_nodes[j]:_cumsum_n_nodes[j+1]] = Y[cumsum_n_nodes[i]:cumsum_n_nodes[i+1]]

        _Y = _Y - a_delta_y_by_y[_Y] + _a_cumsum_n_state[a_perm[a_typ_by_y[_Y]]]
        
        _lE  = [lE[i*n_types+j]  for i in l_perm for j in l_perm]
        _lEF = [lEF[i*n_types+j] for i in l_perm for j in l_perm]
        
        _lX.append( (_lNF, _lE, _lEF) )
        _lY.append(_Y)
    
    if constraints:
        print "WARNING: some constraints are not properly swapped because the node order has a meaning."
        _constraints = list()
        for _lConstraints in constraints:
            for (op, l_l_unary, l_l_state, l_lnegated) in _lConstraints:
                #keep the op but permute by types
                _l_l_unary   = [l_l_unary [i] for i in l_perm]
                _l_l_state   = [l_l_state [i] for i in l_perm]
                _l_lnegated  = [l_lnegated[i] for i in l_perm]
                _lConstraints.append( (op, _l_l_unary, _l_l_state, _l_lnegated))
        _constraints.append(_lConstraints)
        
    return _lX, _lY, _constraints
        
def listConstraints(lX):
    """
    produce the list of constraints for this list of multi-type graphs
    """
    lConstraints = list()
    for _lNF, _lE, _lEF in lX:
        nf_pixel, nf_pict = _lNF
        nb_pixels = len(nf_pixel)
        l_l_unary = [ range(nb_pixels), [0]]
        l_l_states  = [ 0, 0 ]          #we pass a scalar for each type instead of a list since the values are the same across each type
        l_l_negated = [ False, False ]  #same
        
        lConstraint_for_X = [("ANDOUT", l_l_unary, l_l_states, l_l_negated)]  #we have a list of constraints per X
        
        for _state in range(1, NCELL+1):
            lConstraint_for_X.append( ("XOROUT" , l_l_unary
                                            , [ _state, 1 ]      #exactly one cell in state _state with picture label being snake
                                            , l_l_negated) 
                                     ) #we have a list of constraints per X
         
        lConstraints.append( lConstraint_for_X ) 
    return lConstraints


def makeItEasy(lX_pict_feat, lY_pict):
    """
    add the picture label in a feature...
    """
    for X,y in zip(lX_pict_feat, lY_pict):
        X[0] = y


def REPORT(l_Y_GT, lY_Pred, t=None):
    if t: print "\t( predict DONE IN %.1fs)"%t
        
    _flat_GT, _flat_P = (np.hstack([y.ravel() for y in l_Y_GT]),  
                         np.hstack([y.ravel() for y in lY_Pred]))
    confmat = confusion_matrix(_flat_GT, _flat_P)
    print confmat
    print "\ttrace   =", confmat.trace()
    print "\tAccuracy= %.3f"%accuracy_score(_flat_GT, _flat_P)    
    
    
if __name__ == '__main__':
    
    print("Please be patient...")
    snakes = load_snakes()

    #--------------------------------------------------------------------------------------------------
    X_train, Y_train = snakes['X_train'], snakes['Y_train']
    #X_train, Y_train = X_train[:3], Y_train[:3]
    print "TRAIN SET ", len(X_train), len(Y_train)

    if NCELL <10: X_train, Y_train = shorten_snakes(X_train, Y_train, NCELL)

    nb_hidden, X_train, Y_train = augmentWithNoSnakeImages(X_train, Y_train, "train", False, nCell=NCELL)
    print "TRAIN SET ",len(X_train), len(Y_train)
    Y_train_pict = np.array([1]*(len(X_train)-nb_hidden) + [0]*nb_hidden)
    
    X_train = [one_hot_colors(x) for x in X_train]

    X_train, Y_train, Y_train_pict = shuffle_in_unison(X_train, Y_train, Y_train_pict)


    X_train_pict_feat = prepare_picture_data(X_train)
    if bMAKE_PICT_EASY:
        print "Making the train picture task easy"
        makeItEasy(X_train_pict_feat, Y_train_pict)
 
    X_train_directions, X_train_edge_features = prepare_data(X_train)
    #--------------------------------------------------------------------------------------------------
    X_test, Y_test = snakes['X_test'], snakes['Y_test']

    if NCELL <10: X_test, Y_test = shorten_snakes(X_test, Y_test, NCELL)

    nb_hidden, X_test, Y_test = augmentWithNoSnakeImages(X_test, Y_test, "test", False, nCell=NCELL)
    Y_test_pict = np.array([1]*(len(X_test)-nb_hidden) + [0]*nb_hidden)
    print "TEST SET ", len(X_test), len(Y_test)
    
    X_test = [one_hot_colors(x) for x in X_test]

    #useless X_test, Y_test, Y_test_pict = shuffle_in_unison(X_test, Y_test, Y_test_pict)

    X_test_pict_feat = prepare_picture_data(X_test)
    if bMAKE_PICT_EASY:
        print "Making the test picture task easy"
        makeItEasy(X_test_pict_feat, Y_test_pict)    
    
    X_test_directions, X_test_edge_features = prepare_data(X_test)

    #--------------------------------------------------------------------------------------------------
    print "======================================================================================================"
    if True:        
        from pystruct.models.edge_feature_graph_crf import EdgeFeatureGraphCRF
        print "ONE TYPE TRAINING AND TESTING: PIXELS"

#         inference = 'ad3+'
#         inference = 'qpbo'
        inference=INFERENCE
        inference = "qpbo"
        crf = EdgeFeatureGraphCRF(inference_method=inference)
        ssvm = OneSlackSSVM(crf, inference_cache=50, C=.1, tol=.1,  
                            max_iter=MAXITER,
                            n_jobs=N_JOBS
                            #,verbose=1
                            , switch_to='ad3'
                            )

        Y_train_flat = [y_.ravel() for y_ in Y_train]
        print  "\ttrain label histogram : ", np.histogram(np.hstack(Y_train_flat), bins=range(NCELL+2))
        
        t0 = time.time()
        ssvm.fit(X_train_edge_features, Y_train_flat)
        print "FIT DONE IN %.1fs"%(time.time() - t0)
        sys.stdout.flush()
        
        t0 = time.time()
        _Y_pred = ssvm.predict( X_test_edge_features )
        REPORT(Y_test, _Y_pred, time.time() - t0)
    
    #--------------------------------------------------------------------------------------------------
    if True:
        print "_"*50
        print "ONE TYPE TRAINING AND TESTING: PICTURES"
    
        print  "\ttrain label histogram : ", np.histogram(Y_train_pict, bins=range(3))
    
        lr = LogisticRegression(class_weight='balanced')
        
        mdl = GridSearchCV(lr , {'C':[0.1, 0.5, 1.0, 2.0] })
        
        XX = np.vstack(X_train_pict_feat)

        t0 = time.time()
        mdl.fit(XX, Y_train_pict)
        print "FIT DONE IN %.1fs"%(time.time() - t0)
        
        t0 = time.time()
        _Y_pred = mdl.predict( np.vstack(X_test_pict_feat) )
        REPORT([Y_test_pict], _Y_pred, time.time() - t0)
        
    #--------------------------------------------------------------------------------------------------
    print "======================================================================================================"


    # first, train on X with directions only:
    #crf = NodeTypeEdgeFeatureGraphCRF(1, [11], [45], [[2]], inference_method=inference)
    # first, train on X with directions only:
#     l_weights = [ 
#                     [10.0/200] + [10.0/200]*10,
#                     [10.0/20 , 10.0/20]
#             ]
#     print "WEIGHTS:", l_weights
    if nbSWAP_Pixel_Pict_TYPES %2 == 0:
        l_n_states = [NCELL+1, 2]   # 11 states for pixel nodes, 2 states for pictures
        l_n_feat   = [45, 7]        # 45 features for pixels, 7 for pictures
        ll_n_feat  = [[180, 45],    # 2 feature between pixel nodes, 1 between pixel and picture
                      [45  , 0]]   # , nothing between picture nodes (no picture_to_picture edge anyway)
    else:
        l_n_states = [2, NCELL+1]
        l_n_feat   = [7, 45]
        ll_n_feat  = [[0, 45], [45  , 180]]
        
    if not sMODELFILE or not os.path.exists(sMODELFILE):
        print " TRAINING MULTI-TYPE MODEL "
        #TRAINING                              
        crf = NodeTypeEdgeFeatureGraphCRF(2,            # How many node types?
                                          l_n_states,   # How many states per type?
                                          l_n_feat,     # How many node features per type?
                                          ll_n_feat,    # How many edge features per type x type?
                                          inference_method=INFERENCE
    #                                       , l_class_weight = l_weights 
                                          )
        print crf
        
        ssvm = OneSlackSSVM(crf, inference_cache=50, C=.1, tol=0,  
                            max_iter=MAXITER,
                            n_jobs=N_JOBS
                            #,verbose=1
                            #, switch_to='ad3'
                            )
                                                                    
        print "======================================================================================================"
        print "YY[0].shape", Y_train[0].shape
        XX, YY = convertToTwoType(X_train,
                                 X_train_edge_features,    # list of node_feat , edges, edge_feat for pixel nodes
                                 Y_train,
                                 X_train_pict_feat,         #a list of picture_node_features
                                 Y_train_pict,              #a list of integers [0,1]
                                 nCell=NCELL)              
    
        if nbSWAP_Pixel_Pict_TYPES:
            if nbSWAP_Pixel_Pict_TYPES % 2 == 0:
                XX, YY = swap_node_types([1,0], [NCELL+1,       2], XX, YY)
                XX, YY = swap_node_types([1,0], [2      , NCELL+1], XX, YY)
            else:
                XX, YY = swap_node_types([1,0], [NCELL+1,       2], XX, YY)
            
        
        print  "\tlabel histogram : ", np.histogram(   np.hstack([y.ravel() for y in YY]), bins=range(14))
        
        
        print "YY[0].shape", YY[0].shape
        crf.initialize(XX, YY)# check if the data is properly built
        sys.stdout.flush()
        
        t0 = time.time()
        ssvm.fit(XX, YY)
        print "FIT DONE IN %.1fs"%(time.time() - t0)
        sys.stdout.flush()
    
        ssvm.alphas = None  
        ssvm.constraints_ = None
        ssvm.inference_cache_ = None    
        if sMODELFILE:
            print "Saving model in: ", sMODELFILE
            with open(sMODELFILE, "wb") as fd:
                cPickle.dump(ssvm, fd)
    else:
        #REUSE PREVIOUSLY TRAINED MODEL
        print " RUSING PREVIOULSLY TRAINED MULTI-TYPE MODEL: ", sMODELFILE
        
        with open(sMODELFILE, "rb") as fd:
            ssvm = cPickle.load(fd)
    

    print "INFERENCE WITH ", INFERENCE
    XX_test, YY_test =convertToTwoType(X_test,
                     X_test_edge_features,    # list of node_feat , edges, edge_feat for pixel nodes
                     Y_test,
                     X_test_pict_feat,         #a list of picture_node_features
                     Y_test_pict,              #a list of integers [0,1]
                     nCell=NCELL)     
    print  "\tlabel histogram (PIXELs and PICTUREs): ", np.histogram(   np.hstack([y.ravel() for y in YY_test]), bins=range(14))
    
    
    l_constraints = listConstraints(XX_test)
    
    if nbSWAP_Pixel_Pict_TYPES %2 == 1:
        XX_test, YY_test, l_constraints = swap_node_types([1,0], [NCELL+1,       2], XX_test, YY_test, l_constraints)

    print "\t- results without constraints"
    t0 = time.time()
    YY_pred = ssvm.predict( XX_test )
    REPORT(YY_test, YY_pred, time.time() - t0)
    
    print "_"*50
    print "\t- results exploiting constraints"
    t0 = time.time()
    YY_pred = ssvm.predict( XX_test, l_constraints )
    REPORT(YY_test, YY_pred, time.time() - t0)
        
    
    print "_"*50
    
    if INFERENCE == "ad3":
        ssvm.model.inference_method = "ad3+"
    else:
        ssvm.model.inference_method = "ad3"        
    print "INFERENCE WITH ", ssvm.model.inference_method
    t0 = time.time()
    YY_pred = ssvm.predict( XX_test )
    REPORT(YY_test, YY_pred, time.time() - t0)

    print "DONE"
    
    printConfig()


"""

 


"""