"""
==============================================
Conditional Interactions on the Snakes Dataset
==============================================

This is a variant of plot_snakes.py 

Snake are hidding, so we have 2 tasks:
- determining if a snake is in the picture, 
- identifying its head to tail body.  

We use the NodeTypeEdgeFeatureGraphCRF class with 2 type of nodes.

HERE WE GENERATE THE SNAKES AT RANDOM INSTEAD OF USING THE SNAKE DATASET


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

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

from pystruct.learners import OneSlackSSVM
from pystruct.datasets import load_snakes
from pystruct.models import NodeTypeEdgeFeatureGraphCRF

from plot_snakes import one_hot_colors

from plot_hidden_snakes import augmentWithNoSnakeImages, shuffle_in_unison, shorten_snakes

from plot_hidden_short_snakes_typed import plot_snake, prepare_data, prepare_picture_data, convertToTwoType,listConstraints, listConstraints_ATMOSTONE, REPORT

#==============================================================================================

bFIXED_RANDOM_SEED = True  
  
NCELL=10

#INFERENCE="ad3+"    #ad3+ is required when there are hard logic constraints
INFERENCE="ad3"     #ad3 is faster than ad3+ 
N_JOBS=8

#MAXITER=750

lNbSAMPLE=[200, 400, 600, 800]  #how many sample do we generate for each experiment?

nbEXPERIMENT = 10

#==============================================================================================

def printConfig():
    print "== NCELL=", NCELL
    print "== FIXED_SEED=", bFIXED_RANDOM_SEED
    print "== INFERENCE =", INFERENCE
    print "== N_JOBS =", N_JOBS
    #print "== MAX_ITER=", MAXITER
    print "== lNbSAMPLE=", lNbSAMPLE
    print "== nbEXPERIMENT=", nbEXPERIMENT

if __name__ == '__main__': printConfig()


if bFIXED_RANDOM_SEED:
    np.random.seed(1605)
    random.seed(98)
else:
    np.random.seed()
    random.seed()

class GenSnakeException(Exception): pass
    
def genSnakes(N, ncell=NCELL):
    """
    Generate snakes at random.
    Return N tuple (snakes, Y) 
    """
    ltSnakeY = []
    
    ndim = 1+ ncell+1+ncell +1   #where we'll draw each snake.  Border, possible straight snake, centre, possible straight snake, border
    aBoard = np.zeros( (ndim, ndim) , dtype=np.int8)  
    im,jm = 1+ ncell, 1+ ncell #middle of board
    lDirection = range(4) #assume it is N, E, S, W
    lDirectionIncr = [(-1,0), (0,1), (1,0), (0,-1)]
    lDirectionColor = [ [255,0,0], [255,255,0], [0,255,0], [0,255,255] ]
    for _n in range(N):
        while True:
            aBoard[:,:] = -1 #all background
            i,j = im,jm
            lij = list()
            ldir=list()
            aSnake, Y = None, None
            
            try:
                for _ncell in range(ncell):
                    random.shuffle(lDirection) #we will try each direction in turn
                    for dir in lDirection:
                        _i, _j = i+lDirectionIncr[dir][0], j+lDirectionIncr[dir][1]
                        if aBoard[_i,_j] == -1: break   #ok, valid direction, we jump on a background pixel
                    if aBoard[_i,_j] != -1: raise GenSnakeException("Failed to generate a snake") #got stuck
                    aBoard[i,j] = dir
                    lij.append( (i,j) )
                    ldir.append(dir)
                    i,j = _i,_j
                #ok we have a Snake, let's create the image with background borders
                imin,jmin = map(min, zip(*lij))
                imax,jmax = map(max, zip(*lij))
                aSnake = np.zeros((imax-imin+3, jmax-jmin+3, 3), dtype=np.uint8)
                aSnake[:,:,2] = 255  #0,0,255
                aY     = np.zeros((imax-imin+3, jmax-jmin+3)   , dtype=np.uint8)
                for _lbl, ((_i,_j), _dir) in enumerate(zip(lij, ldir)): 
                    aSnake[_i-imin+1, _j-jmin+1,:] = lDirectionColor[_dir]
                    aY    [_i-imin+1, _j-jmin+1]   = _lbl + 1
                    
                break 
            except GenSnakeException: pass
        ltSnakeY.append( (aSnake, aY) )
#         print aSnake
#         print aY
#         plot_snake(aSnake)
    return ltSnakeY


if __name__ == '__main__':

        
    print("Please be patient...")
    snakes = load_snakes()

    #--------------------------------------------------------------------------------------------------
    #we always test against the original test set
    X_test, Y_test = snakes['X_test'], snakes['Y_test']
    if NCELL <10: X_test, Y_test = shorten_snakes(X_test, Y_test, NCELL)

    nb_hidden, X_test, Y_test = augmentWithNoSnakeImages(X_test, Y_test, "test", False, nCell=NCELL)
    Y_test_pict = np.array([1]*(len(X_test)-nb_hidden) + [0]*nb_hidden)
    print "TEST SET ", len(X_test), len(Y_test)
    
    X_test = [one_hot_colors(x) for x in X_test]
    X_test_pict_feat = prepare_picture_data(X_test)
    X_test_directions, X_test_edge_features = prepare_data(X_test)

    #--------------------------------------------------------------------------------------------------
    for iExp in range(nbEXPERIMENT):
        print "#"*75
        print "# EXPERIMENT %d / %d"%(iExp+1, nbEXPERIMENT)
        print "#"*75
        
        
        lXY = genSnakes(max(lNbSAMPLE))
        X_train_all, Y_train_all = zip(*lXY)
        X_train_all, Y_train_all = list(X_train_all), list(Y_train_all)
        print "*****  GENERATED %d snakes of length %d *****"%(len(X_train_all), NCELL)
    
        #Also generate an additional test set
        NTEST=100
        lXYTest = genSnakes( NTEST )
        X_test_gen, Y_test_gen = zip(*lXYTest)
        X_test_gen, Y_test_gen = list(X_test_gen), list(Y_test_gen)
        print "*****  GENERATED %d snakes of length %d *****"%(NTEST, NCELL)
        nb_hidden, X_test_gen, Y_test_gen = augmentWithNoSnakeImages(X_test_gen, Y_test_gen, "test_gen", False, nCell=NCELL)
        Y_test_gen_pict = np.array([1]*(len(X_test_gen)-nb_hidden) + [0]*nb_hidden)
        print "GENERATED TEST SET ", len(X_test_gen), len(Y_test_gen)
        
        
        X_test_gen = [one_hot_colors(x) for x in X_test_gen]
        X_test_gen_pict_feat = prepare_picture_data(X_test_gen)
        X_test_gen_directions, X_test_gen_edge_features = prepare_data(X_test_gen)
    
        for nbSample in lNbSAMPLE:
            print "======================================================================================================"
            print "TRAINING"
            X_train, Y_train = X_train_all[0:nbSample], Y_train_all[0:nbSample]
    
            nb_hidden, X_train, Y_train = augmentWithNoSnakeImages(X_train, Y_train, "train", False, nCell=NCELL)
            print "TRAIN SET ",len(X_train), len(Y_train)
            Y_train_pict = np.array([1]*(len(X_train)-nb_hidden) + [0]*nb_hidden)
            
            X_train = [one_hot_colors(x) for x in X_train]
            X_train, Y_train, Y_train_pict = shuffle_in_unison(X_train, Y_train, Y_train_pict)
            X_train_pict_feat = prepare_picture_data(X_train)
            X_train_directions, X_train_edge_features = prepare_data(X_train)
                
            #--------------------------------------------------------------------------------------------------
            if True:        
                print "==========================================================================="
                from pystruct.models.edge_feature_graph_crf import EdgeFeatureGraphCRF
                print "ONE TYPE TRAINING AND TESTING: PIXELS"
        
                inference = "qpbo"
                crf = EdgeFeatureGraphCRF(inference_method=inference)
                ssvm = OneSlackSSVM(crf, inference_cache=50, C=.1, tol=.1,  
                                    #max_iter=MAXITER,
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
                REPORT(Y_test, _Y_pred, time.time() - t0, NCELL, "gen.csv", True, "singletype_%d"%nbSample)
                _Y_pred = ssvm.predict( X_test_gen_edge_features )
                REPORT(Y_test_gen, _Y_pred, None        , NCELL, "gen.csv", True, "singletype_%d_gentest"%nbSample)
            
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
                REPORT([Y_test_pict], _Y_pred, time.time() - t0, 2, "gen.csv", True, "picture_logit_%d"%nbSample)
                
            #--------------------------------------------------------------------------------------------------
            print "======================================================================================================"
        
            l_n_states = [NCELL+1, 2]   # 11 states for pixel nodes, 2 states for pictures
            l_n_feat   = [45, 7]        # 45 features for pixels, 7 for pictures
            ll_n_feat  = [[180, 45],    # 2 feature between pixel nodes, 1 between pixel and picture
                          [45  , 0]]   # , nothing between picture nodes (no picture_to_picture edge anyway)
                
            print " TRAINING MULTI-TYPE MODEL "
            #TRAINING                              
            crf = NodeTypeEdgeFeatureGraphCRF(2,            # How many node types?
                                              l_n_states,   # How many states per type?
                                              l_n_feat,     # How many node features per type?
                                              ll_n_feat,    # How many edge features per type x type?
                                              inference_method=INFERENCE
                                              )
            print crf
            ssvm = OneSlackSSVM(crf, inference_cache=50, C=.1, tol=0.1,  
                                #max_iter=MAXITER,
                                n_jobs=N_JOBS
                                )
                                                                        
            print "======================================================================================================"
            print "YY[0].shape", Y_train[0].shape
            XX, YY = convertToTwoType(X_train,
                                     X_train_edge_features,    # list of node_feat , edges, edge_feat for pixel nodes
                                     Y_train,
                                     X_train_pict_feat,         #a list of picture_node_features
                                     Y_train_pict,              #a list of integers [0,1]
                                     nCell=NCELL)              
        
            print  "\tlabel histogram : ", np.histogram(   np.hstack([y.ravel() for y in YY]), bins=range(14))
            
            
            print "YY[0].shape", YY[0].shape
            crf.initialize(XX, YY)# check if the data is properly built
            sys.stdout.flush()
            
            t0 = time.time()
            ssvm.fit(XX, YY)
            print "FIT DONE IN %.1fs"%(time.time() - t0)
            sys.stdout.flush()
            
            print "_"*50
            XX_test, YY_test =convertToTwoType(X_test,
                             X_test_edge_features,    # list of node_feat , edges, edge_feat for pixel nodes
                             Y_test,
                             X_test_pict_feat,         #a list of picture_node_features
                             Y_test_pict,              #a list of integers [0,1]
                             nCell=NCELL)     
            print  "\tlabel histogram (PIXELs and PICTUREs): ", np.histogram(   np.hstack([y.ravel() for y in YY_test]), bins=range(14))
            XX_test_gen, YY_test_gen =convertToTwoType(X_test_gen,
                             X_test_gen_edge_features,    # list of node_feat , edges, edge_feat for pixel nodes
                             Y_test_gen,
                             X_test_pict_feat,         #a list of picture_node_features
                             Y_test_pict,              #a list of integers [0,1]
                             nCell=NCELL)     
            
            
            l_constraints     = listConstraints_ATMOSTONE(XX_test    , NCELL)
            l_constraints_gen = listConstraints_ATMOSTONE(XX_test_gen, NCELL)
            
            print "_"*50
            print "\t- results without constraints (using %s)"%INFERENCE
            t0 = time.time()
            YY_pred = ssvm.predict( XX_test )
            REPORT(YY_test, YY_pred, time.time() - t0       , NCELL+2, "gen.csv", True, "multitype_%d"%nbSample)
            YY_pred = ssvm.predict( XX_test_gen )
            REPORT(YY_test_gen, YY_pred, None               , NCELL+2, "gen.csv", True, "multitype_%d_gentest"%nbSample)
            
            print "_"*50
            print "\t- results exploiting constraints (using ad3+)"
            ssvm.model.inference_method = "ad3+"
            t0 = time.time()
            YY_pred = ssvm.predict( XX_test     , l_constraints )
            REPORT(YY_test, YY_pred, time.time() - t0    , NCELL+2, "gen.csv", True, "multitype_constraints_%d"%nbSample)
            YY_pred = ssvm.predict( XX_test_gen , l_constraints_gen )
            REPORT(YY_test_gen, YY_pred, None            , NCELL+2, "gen.csv", True, "multitype_constraints_%d_gentest"%nbSample)
                
            
            print "_"*50
            
            print "One Experiment DONE"

        print "ALL EXPERIMENTS DONE"
        
        printConfig()
 