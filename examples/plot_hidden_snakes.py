"""
==============================================
Conditional Interactions on the Snakes Dataset
==============================================

This is a variant of plot_snakes.py 

Snake are hiding!! Therefore, some picture have colored pixels despite they do not contain any snake.

    JL Meunier - January 2017
    
    Developed for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943
    
    Copyright Xerox

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
"""
from __future__ import (absolute_import, division, print_function)

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, accuracy_score
import time

from pystruct.learners import OneSlackSSVM
from pystruct.datasets import load_snakes
from pystruct.models import EdgeFeatureGraphCRF
from pystruct.models import NodeTypeEdgeFeatureGraphCRF

from plot_snakes import  one_hot_colors, prepare_data


def isSnakePresent(a_hot_picture, nCell=10):
    """
    Algorithmic check, to make sure that after tempering with the snake we do not have a snake! :-)
    Works on the 1-hot encoded picture
    """
    ai, aj = np.where(a_hot_picture[...,3] != 1)
    
    #let's start from each cell until we can walk thru an entire snake
    #yeah, brute force, but otherwise it is tricky to check!!
    bSnake = False
    for i0,j0 in zip(ai,aj):
    
        lij  = walkThruSnake(a_hot_picture, (i0, j0), nCell)
        if len(lij) == nCell-1:
            bSnake = True
            break 
    return bSnake
    
def walkThruSnake(a_hot_picture, tIJ, nCell=10):
    """
    Walk thru the snake from I,J
    Return the list of visited cells (excluding start cell)
    """
    (i,j) = tIJ
    lij = list()
    color_index = np.where(a_hot_picture[i,j,:]==1)[0][0]
    while len(lij) < nCell -1:
        dj = np.array( [ 0, 0, 1, None, -1])[color_index]
        di = np.array( [-1, 1, 0, None,  0])[color_index]
        i += di
        j += dj
        color_index = np.where(a_hot_picture[i,j,:]==1)[0][0]
        if color_index == 3: break  #background
        if (i,j) in lij: break      #crossing itself, or looping
        lij.append((i,j))
    return lij
    
def changeOneSnakeCell(a_picture, bOneHot=True, nCell=10): #in place!!
    """
    Change the color of 1 snake cells into another snake cell color
    """
    if bOneHot:
        ai, aj = np.where(a_picture[...,3] != 1)
    else:
        _p = np.copy(a_picture)
        _p = one_hot_colors(_p)
        ai, aj = np.where(_p[...,3] != 1)
    assert len(ai) == nCell, (len(ai), nCell)
    
    iChange = random.randint(0,nCell-1)
    
    for i in range(10):
        iFromCell = random.randint(0,nCell-1)
        if  (a_picture[ai[iChange], aj[iChange],:] != a_picture[ai[iFromCell], aj[iFromCell],:]).any():
            a_picture[ai[iChange], aj[iChange],:]  = a_picture[ai[iFromCell], aj[iFromCell],:]
            #so that we do not care about which color is valid...
            break

    return a_picture

def distortSnake(a_picture, bOneHot=True, nCell=10):
    """
    Shuffle either the snake's cells or the pcitures' pixels.
    """
    bDOCUMENT = False   #to show the change on screen
    
    if bDOCUMENT:  
        pict_mem = np.copy(a_picture)
    
    changeOneSnakeCell(a_picture, bOneHot, nCell=nCell)
    
    if bDOCUMENT:  
        if bOneHot:
            zz = a_picture
        else:
            zz = one_hot_colors(a_picture)
        if not isSnakePresent(zz, nCell):
            plot_snake(pict_mem)
            plot_snake(a_picture)
    
def convertToSingleTypeX(X):
    """
    For NodeTypeEdgeFeatureGraphCRF X is structured differently.
    But NodeTypeEdgeFeatureGraphCRF can handle graphs with a single node type. One simply needs to convert X to the new structure using this method.
    """
    return [([nf], [e], [ef]) for (nf,e,ef) in X]


def plot_snake(picture):
    plt.imshow(picture, interpolation='nearest')
    plt.show()


def augmentWithNoSnakeImages(X,Y, name, bOneHot=True, iMult=1, nCell=10):
    """
    return the number of added picture (ADDED AT THE END OF INPUT LISTS)
    """
    print("ADDING PICTURE WIHOUT SNAKES!!!   %d elements in %s"%(len(X), name))

    X_NoSnake = []
    Y_NoSnake = []
    for i in range(int(iMult)):
        X_NoSnake.extend([np.copy(x) for x in X])
        Y_NoSnake.extend([np.copy(y) for y in Y]) #shorten_sakes does modify Y...
        
    if True:
        #best method for our experiment
        for x in X_NoSnake: distortSnake(x, bOneHot, nCell)
    else:
        shorten_snakes(X_NoSnake, Y_NoSnake, nCell-1)
    
    newX = list()
    newY = list()
    for x,y in zip(X_NoSnake, Y_NoSnake):
        _x = x if bOneHot else one_hot_colors(x)
        if isSnakePresent(_x):
            print("\t- DISCARDING a shuffled snake which is still a snake!!!!")
#             if True and not bOneHot: plot_snake(x)
        else:
            newX.append(x)
            newY.append(np.zeros(y.shape, dtype=np.int32))
    assert len(newX)==len(newY)
    return len(newX), X+newX, Y+newY
    
def shuffle_in_unison(*args):    
    lTuple = list(zip(*args))
    random.shuffle(lTuple)
    return zip(*lTuple)

def shorten_snakes(lX,lY, N):
    """
    It is faster to work on shorter snakes, but easier as well for the models 
    """
    newlX,newlY = list(), list()
    for X, Y in zip(lX,lY):
        assert X.shape[:2] == Y.shape, (X.shape, Y.shape)
        ai, aj = np.where(Y>N)
        X[ai,aj,:] = X[0,0,:]
        Y[ai,aj] = 0
        #crop
        ai, aj = np.where(Y!=0)
        aimin,aimax = min(ai)-1, max(ai)+2
        ajmin,ajmax = min(aj)-1, max(aj)+2
        newlY.append( Y[aimin:aimax, ajmin:ajmax] )
        newlX.append( X[aimin:aimax, ajmin:ajmax,:])
        
    return newlX, newlY

#=====================================================================================================
if __name__ == '__main__':
    np.random.seed(1605)
    random.seed(98)

    print("Please be patient. Learning will take 5-20 minutes.")
    
    #if you want to shorten all the snakes
    #NCELL = 3
    NCELL = 10
    print("NCELL=", NCELL)
    
    
    snakes = load_snakes()

    # --- TRAIN
    X_train, Y_train = snakes['X_train'], snakes['Y_train']
    #X_train, Y_train = X_train[:10], Y_train[:10]    #if you want to debug...
    if NCELL < 10: X_train, Y_train = shorten_snakes(X_train, Y_train, NCELL)

    nbNoSnake, X_train, Y_train = augmentWithNoSnakeImages(X_train, Y_train, "train", bOneHot=False, nCell=NCELL)
    X_train = [one_hot_colors(x) for x in X_train]
    X_train, Y_train = shuffle_in_unison(X_train, Y_train)
    X_train_directions, X_train_edge_features = prepare_data(X_train)
    Y_train_flat = [y_.ravel() for y_ in Y_train]

    print("%d picture for training"%len(X_train))

    # --- TEST
    X_test, Y_test = snakes['X_test'], snakes['Y_test']
    if NCELL < 10: X_test, Y_test = shorten_snakes(X_test, Y_test, NCELL)
    _, X_test, Y_test = augmentWithNoSnakeImages(X_test, Y_test, "test", bOneHot=False, nCell=NCELL)
    
    X_test = [one_hot_colors(x) for x in X_test]
    X_test_directions, X_test_edge_features = prepare_data(X_test)
    Y_test_flat = [y_.ravel() for y_ in Y_test]

    print("%d picture for test"%len(X_test))    

    # -------------------------------------------------------------------------------------        

    inference = 'qpbo'
    bClassic = True  #True => use the old good EdgeFeatureGraphCRF

    # now, use more informative edge features:
    t0 = time.time()
    if bClassic:
        print("EdgeFeatureGraphCRF")
        crf = EdgeFeatureGraphCRF(inference_method=inference)
        ssvm = OneSlackSSVM(crf, inference_cache=50, C=.1, tol=.1, 
                        #WHY THIS??? max_iter=100,
                        #why not this switch_to=ad3??? 
                        switch_to='ad3',
                        #verbose=1,
                        n_jobs=2,
                        )
        ssvm.fit( X_train_edge_features , Y_train_flat)
    else:    
        print("NodeTypeEdgeFeatureGraphCRF")
        crf = NodeTypeEdgeFeatureGraphCRF(1, [11], [45], [[180]], inference_method=inference)
        ssvm = OneSlackSSVM(crf, inference_cache=50, C=.1, tol=.1, 
                        switch_to='ad3',
                        #JL adds a max-iter sometimes
                        #max_iter=100,
                        n_jobs=1)
        ssvm.fit( convertToSingleTypeX(X_train_edge_features) , Y_train_flat)
    print("Training time = %.1fs"%(time.time()-t0))
    
    if bClassic:    
        Y_pred2 = ssvm.predict( X_test_edge_features )
    else:
        Y_pred2 = ssvm.predict( convertToSingleTypeX(X_test_edge_features) )
    print("Results using input features for edges")
    print("Test accuracy: %.3f"
          % accuracy_score(np.hstack(Y_test_flat), np.hstack(Y_pred2)))
    print(confusion_matrix(np.hstack(Y_test_flat), np.hstack(Y_pred2)))
    


    #------------------------------------------------------------------------------------------------------------------------
    #Predict under constraints
    if True and not bClassic:
        def buildConstraintsFromSingleTyped(X, bOne=True):
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
            for ([nf], [e], [ef]) in X:
                n_nodes = nf.shape[0]
                lConstraintPerGraph = [ (sLogicOp, range(n_nodes), i, False) for i in range(1,NCELL+1) ] #only one 
                lConstraint.append( lConstraintPerGraph )
            return lConstraint
        
        X_3 = convertToSingleTypeX(X_test_edge_features)
        lC = buildConstraintsFromSingleTyped(X_3, False)
        Y_pred2 = ssvm.predict( X_3, lC )
        print("Results using also input features for edges")
        print("Inference with an ATMOST constraint per snake label")
        print("Test accuracy: %.3f"
              % accuracy_score(np.hstack(Y_test_flat), np.hstack(Y_pred2)))
        print(confusion_matrix(np.hstack(Y_test_flat), np.hstack(Y_pred2)))        

