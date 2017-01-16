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
"""
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, accuracy_score
import time

from pystruct.learners import OneSlackSSVM
from pystruct.datasets import load_snakes
from pystruct.utils import make_grid_edges, edge_list_to_features
#from pystruct.models import EdgeFeatureGraphCRF
from pystruct.models import NodeTypeEdgeFeatureGraphCRF

from plot_snakes import  one_hot_colors, neighborhood_feature, prepare_data

def isSnakePresent(a_hot_picture):
    """
    Algorithmic check, to make sure that after shuffling we do not have a snake! :-)
    work on the 1-hot encoded picture
    """
    try:
        ai, aj = np.where(a_hot_picture[...,3] != 1)
        if len(ai) != 10: return False
        lij = zip(ai, aj)
        for n in range(10):
            _lij = shiftSnake(a_hot_picture, lij)
            if len(_lij) != len(lij)-1: return False
            lij = _lij
        if len(_lij) != 0: return False
        return True
    except:
        return False
    
def shiftSnake(a_hot_picture, lij):
    #the snake moves by one cell, head disappearing in sand
    _lij = list()
    for i,j in lij:
        color_index = np.where(a_hot_picture[i,j,:]==1)[0][0]
        dj = np.array( [ 0, 0, 1, None, -1])[color_index]
        di = np.array( [-1, 1, 0, None,  0])[color_index]
        i,j = i+di,j+dj
        if a_hot_picture[i,j,3] != 1: #backgroun
            _lij.append((i,j))
    return _lij

def shufflePictureCells(a_picture):  #in place!!
    """
    Shuffle the pixels
    """
    n = random.randint(1,4)
    if n == 1:
        map(np.random.shuffle, a_picture)
    elif n == 2:
        map(np.random.shuffle, np.transpose(a_picture, (1,0,2)))
    else:
        map(np.random.shuffle, a_picture)
        map(np.random.shuffle, np.transpose(a_picture, (1,0,2)))
        
    return a_picture
        
def shuffleSnakeCells(a_picture, bOneHot=True): #in place!!
    """
    Shuffle the colors of the 10 snake cells
    """
    if bOneHot:
        ai, aj = np.where(a_picture[...,3] != 1)
    else:
        _p = np.copy(a_picture)
        _p = one_hot_colors(_p)
        ai, aj = np.where(_p[...,3] != 1)
    assert len(ai) == 10
    
    l_shuffled_aij = zip(ai,aj)
    random.shuffle( l_shuffled_aij )
    _ai, _aj = zip(*l_shuffled_aij)
    
    a_picture[_ai,_aj,:] =  a_picture[ai,aj,:]
    return a_picture

def changeOneSnakeCell(a_picture, bOneHot=True): #in place!!
    """
    Change the color of 1 snake cells
    """
    if bOneHot:
        ai, aj = np.where(a_picture[...,3] != 1)
    else:
        _p = np.copy(a_picture)
        _p = one_hot_colors(_p)
        ai, aj = np.where(_p[...,3] != 1)
    assert len(ai) == 10
    
    iChange = random.randint(0,9)
    
    while True:
        iFromCell = random.randint(0,9)
        if  (a_picture[ai[iChange], aj[iChange],:] != a_picture[ai[iFromCell], aj[iFromCell],:]).any():
             a_picture[ai[iChange], aj[iChange],:]  = a_picture[ai[iFromCell], aj[iFromCell],:]
             #so that we do not care about which color is valid...
             break

    return a_picture

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
    
def convertToSingleTypeX(X):
    """
    For NodeTypeEdgeFeatureGraphCRF X is structured differently.
    But NodeTypeEdgeFeatureGraphCRF can handle graph with a single node type. One needs to convert X to the new structure using this method.
    """
    return [([nf], [e], [ef]) for (nf,e,ef) in X]

def plot_snake(picture):
    plt.imshow(picture, interpolation='nearest')
    plt.show()

def augmentWithNoSnakeImages(X,Y, name, bOneHot=True):
    print "ADDING PICTURE WIHOUT SNAKES!!!   %d elements in %s"%(len(X), name)
    
    X_NoSnake = [np.copy(x) for x in X]
    for x in X_NoSnake: shuffleSnake(x, bOneHot)
    #map(shufflePictureCells, X_NoSnake)
    
    newX = list()
    Y_NoSnake = list()
    for x,y in zip(X_NoSnake, Y):
        if isSnakePresent(x):
            print "\t- DISCARDING a shuffled snake which is still a snake!!!!"
        else:
            newX.append(x)
            Y_NoSnake.append(np.zeros(y.shape, dtype=np.int8))
    X_NoSnake = newX
    
    return X+X_NoSnake, Y+Y_NoSnake
    
def shuffle_XY(X,Y):    
    lxy = zip(X, Y)
    random.shuffle(lxy)
    X, Y = zip(*lxy)
    return X, Y

if __name__ == '__main__':
    print("Please be patient. Learning will take 5-20 minutes.")
    snakes = load_snakes()
    X_train, Y_train = snakes['X_train'], snakes['Y_train']
    
    bSHUFFLE = True
    
    bADD_HIDDEN_SNAKES = True
    #bADD_HIDDEN_SNAKES = False
    #JL
    #X_train, Y_train = X_train[:10], Y_train[:10]
    print len(X_train), len(Y_train)
    #print `X_train[0]`

    if bADD_HIDDEN_SNAKES:
        X_train, Y_train = augmentWithNoSnakeImages(X_train, Y_train, "train", False)
        print len(X_train), len(Y_train)
    
    if False:
        #show the faked pictures
        for ix, x in enumerate(X_train): plot_snake(shufflePictureCells(x))  

    X_train_hot = [one_hot_colors(x) for x in X_train]
    
    if False:
        for ix, x in enumerate(X_train_hot): 
            if not isSnakePresent(x): plot_snake(X_train[ix])
    
    X_train = X_train_hot
    print "Snakes are ok"
    

    if bSHUFFLE:
        #let's shuffle our data
        X_train, Y_train = shuffle_XY(X_train, Y_train)

    # -------------------------------------------------------------------------------------        
    X_train_directions, X_train_edge_features = prepare_data(X_train)

    Y_train_flat = [y_.ravel() for y_ in Y_train]

    inference = 'qpbo'
    # first, train on X with directions only:
    #CHANGE!!
    #We require NodeTypeEdgeFeatureGraphCRF
    #crf = NodeTypeEdgeFeatureGraphCRF(inference_method=inference)
    crf = NodeTypeEdgeFeatureGraphCRF(1, [11], [45], [[2]], inference_method=inference)
    XX = convertToSingleTypeX(X_train_directions)
    ssvm = OneSlackSSVM(crf, inference_cache=50, C=.1, tol=.1, 
                        max_iter=100,
                        n_jobs=1)
    print len(XX), len(Y_train), len(Y_train_flat)
    ssvm.fit(XX, Y_train_flat)
    
    # Evaluate using confusion matrix.
    # Clearly the middel of the snake is the hardest part.
    X_test, Y_test = snakes['X_test'], snakes['Y_test']
    print "TEST len=", len(X_test)
    if bADD_HIDDEN_SNAKES:
        X_test, Y_test = augmentWithNoSnakeImages(X_test, Y_test, "test", bOneHot=False)
        print "TEST len=", len(X_test)
    
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
    ssvm = OneSlackSSVM(crf, inference_cache=50, C=.1, tol=.1, 
                        switch_to='ad3',
                        #JL adds a max-iter sometimes
                        #max_iter=100,
                        n_jobs=1)
    t0 = time.time()
    ssvm.fit( convertToSingleTypeX(X_train_edge_features) , Y_train_flat)
    print "Training time = %.1fs"%(time.time()-t0)
    
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

 


"""