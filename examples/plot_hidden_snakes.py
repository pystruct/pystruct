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
    """
    return the number of added picture (AT THE END OF INPUT LISTS)
    """
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
    
    return len(X_NoSnake), X+X_NoSnake, Y+Y_NoSnake
    
def shuffle_in_unison(*args):    
    lTuple = zip(*args)
    random.shuffle(lTuple)
    return zip(*lTuple)

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
        _, X_train, Y_train = augmentWithNoSnakeImages(X_train, Y_train, "train", False)
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
        _, X_test, Y_test = augmentWithNoSnakeImages(X_test, Y_test, "test", bOneHot=False)
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
----------------------------------------------------------------
ALWAYS SHUFFLING!!

WITHOUT HIDDEN SNAKES
 
Please be patient. Learning will take 5-20 minutes.
200 200
Snakes are ok
200 200 200
TEST len= 100
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
Training time = 37.5s
Results using also input features for edges
Test accuracy: 0.907
[[2749    0    0    0    0    0    0    0    1    0    0]
 [   0   99    0    1    0    0    0    0    0    0    0]
 [   0    0   98    0    1    0    0    1    0    0    0]
 [   0    9    2   79    1    6    0    2    1    0    0]
 [   0    1   38    4   38    0   15    2    2    0    0]
 [   1    5    3   41    2   30    1   13    1    3    0]
 [   1    0   17    7   12    1   44    1   15    0    2]
 [   1    3    1   19    5    7    2   52    2    8    0]
 [   0    2   10    1    9    2    4    2   63    1    6]
 [   2    0    2   14    0    5    0    3    2   71    1]
 [   1    0    2    2   12    0    5    0    1    0   77]]

 --------------------------
    switch_to='ad3',
    max-iter=100
    
 Results using also input features for edges
Test accuracy: 0.870
[[2750    0    0    0    0    0    0    0    0    0    0]
 [   1   94    2    1    0    0    1    0    0    1    0]
 [   0    7   88    0    0    0    2    0    2    0    1]
 [   0   30   11   41    0    7    0    9    0    2    0]
 [   4    6   38   11   17    3    7    0   13    0    1]
 [   2    9   10   25    4   24    3   13    2    8    0]
 [   0    9   18    9    8    6   23    1   19    1    6]
 [   2    9    9   12    6   10    4   34    2   11    1]
 [   0    8   13    3    4    3    4    1   54    2    8]
 [  10    8    6    6    1    3    1    4    2   57    2]
 [   1    3    3    4    4    0    0    0    6    0   79]]


--------------------------
    switch_to='ad3',
    without max_iter
    
Results using also input features for edges
Test accuracy: 0.997
[[2749    0    0    0    0    0    0    0    1    0    0]
 [   0  100    0    0    0    0    0    0    0    0    0]
 [   0    0  100    0    0    0    0    0    0    0    0]
 [   0    0    0   99    0    0    0    0    0    1    0]
 [   0    0    0    0   99    0    1    0    0    0    0]
 [   0    0    0    1    0   98    0    1    0    0    0]
 [   0    0    0    0    1    0   98    0    1    0    0]
 [   0    0    0    0    0    1    0   99    0    0    0]
 [   0    0    0    0    0    0    0    0  100    0    0]
 [   0    0    0    1    0    0    0    1    0   98    0]
 [   0    0    0    0    1    0    0    0    0    0   99]]

----------------------------------------------------------------
 
SHUFFLING EITHER PIXELS OR SNAKE CELLS

Please be patient. Learning will take 5-20 minutes.
200 200
ADDING PICTURE WIHOUT SNAKES!!!   200 elements in train
400 400
Snakes are ok
400 400 400
TEST len= 100
ADDING PICTURE WIHOUT SNAKES!!!   100 elements in test
TEST len= 200
Results using only directional features for edges
Test accuracy: 0.858
[[6336    2    1    6    7   11   13   72   31   11   10]
 [  87    9    0    0    2    0    0    0    0    2    0]
 [  44    0    2    1    1    1    9   31   10    1    0]
 [  49    0    0    1    0    6    5   15   18    3    3]
 [  50    0    1    0    2    2   12   16   12    2    3]
 [  52    1    0    2    0    2   11   23    4    5    0]
 [  58    0    1    0    3    1   13   14    7    2    1]
 [  57    0    1    1    1    5    1   16   10    5    3]
 [  57    1    0    1    3    3    8    8   12    3    4]
 [  57    0    0    0    1    0    1   16    8   15    2]
 [  58    0    0    0    0    3    0    9    3    3   24]]
Training time = 44.0s
Results using also input features for edges
Test accuracy: 0.864
[[6439    1    0    4    8    5    6    7    1   10   19]
 [  98    1    0    1    0    0    0    0    0    0    0]
 [  98    0    1    1    0    0    0    0    0    0    0]
 [  98    0    0    2    0    0    0    0    0    0    0]
 [  98    0    0    0    0    0    2    0    0    0    0]
 [  95    0    0    0    0    5    0    0    0    0    0]
 [  95    0    0    0    0    0    5    0    0    0    0]
 [  95    0    0    0    0    0    0    5    0    0    0]
 [  94    0    0    0    0    0    0    0    5    0    1]
 [  94    0    0    0    0    0    0    0    0    6    0]
 [  91    0    0    0    1    0    0    0    0    0    8]]


 --------------------------
    switch_to='ad3',
     max-iter=100
 
Training time = 34.5s
Results using also input features for edges
Test accuracy: 0.870
[[6384    2    0    0    1   13   11    6    4   20   59]
 [  92    5    1    0    1    0    0    0    0    0    1]
 [  86    0    4    1    0    3    1    0    0    2    3]
 [  80    1    1    4    2    4    2    1    3    0    2]
 [  79    0    1    3    3    3    3    2    2    4    0]
 [  74    0    1    0    2    8    2    5    3    1    4]
 [  69    0    0    2    0    4   10    1    8    4    2]
 [  66    0    0    0    2    0    2   11    3   11    5]
 [  58    0    0    0    1    2    0    3   23    2   11]
 [  57    0    0    0    0    1    1    2    0   34    5]
 [  57    0    0    0    0    0    0    1    0    0   42]]
 --------------------------
    switch_to='ad3',
    without max-iter

Training time = 1346.7s
Results using also input features for edges
Test accuracy: 0.987
[[6437    7    8    8    4    2    1    0    7   14   12]
 [   2   97    0    0    0    1    0    0    0    0    0]
 [   2    0   97    0    1    0    0    0    0    0    0]
 [   0    0    0   97    0    2    0    1    0    0    0]
 [   0    0    1    0   96    0    2    0    1    0    0]
 [   0    0    0    2    0   95    0    3    0    0    0]
 [   0    0    1    0    2    0   94    0    3    0    0]
 [   0    0    0    1    0    3    0   93    0    3    0]
 [   0    0    1    0    1    0    1    0   97    0    0]
 [   0    0    0    0    0    1    0    1    0   98    0]
 [   0    0    0    0    0    0    1    0    1    0   98]]

 
 
----------------------------------------------------------------
CHANGING ONE CELL OF THE SNAKE 
    switch_to='ad3',
    without max-iter
 
 Please be patient. Learning will take 5-20 minutes.
200 200
ADDING PICTURE WIHOUT SNAKES!!!   200 elements in train
400 400
Snakes are ok
400 400 400
TEST len= 100
ADDING PICTURE WIHOUT SNAKES!!!   100 elements in test
TEST len= 200
Results using only directional features for edges
Test accuracy: 0.857
[[6355    0    0    0    4   26    0    8    1    4  102]
 [ 100    0    0    0    0    0    0    0    0    0    0]
 [  91    0    0    0    0    9    0    0    0    0    0]
 [  91    0    0    0    0    0    0    0    0    0    9]
 [  99    0    0    0    0    0    0    0    0    0    1]
 [  96    0    0    0    0    1    0    1    0    0    2]
 [  97    0    0    0    1    0    0    0    1    0    1]
 [  95    0    0    0    0    4    0    1    0    0    0]
 [  86    0    0    0    2    0    0    0    1    0   11]
 [  70    0    0    0    0   13    0    3    0    7    7]
 [  34    0    0    0    0    0    0    2    0    0   64]]
Training time = 1852.6s
Results using also input features for edges
Test accuracy: 0.904
[[6185   25   25   25   25   24   25   32   39   42   53]
 [  41   58    0    0    0    0    1    0    0    0    0]
 [  41    0   56    0    2    0    0    1    0    0    0]
 [  41    0    1   56    0    2    0    0    0    0    0]
 [  39    0    0    1   56    0    4    0    0    0    0]
 [  39    0    0    0    1   58    0    2    0    0    0]
 [  39    0    0    0    0    1   59    0    1    0    0]
 [  38    0    0    0    0    0    1   60    0    1    0]
 [  36    1    0    0    0    0    0    0   62    0    1]
 [  36    0    0    1    1    0    0    0    0   62    0]
 [  32    1    0    0    1    1    0    0    0    0   65]]


----------------------------------------------------------------
CHANGING TWO CELLs OF THE SNAKE 
    switch_to='ad3',
    without max-iter
 
Please be patient. Learning will take 5-20 minutes.
200 200
ADDING PICTURE WIHOUT SNAKES!!!   200 elements in train
400 400
Snakes are ok
400 400 400
TEST len= 100
ADDING PICTURE WIHOUT SNAKES!!!   100 elements in test
TEST len= 200
Results using only directional features for edges
Test accuracy: 0.853
[[6318    5   13    8    5    9   26   18   25   30   43]
 [  93    5    0    0    0    1    0    0    0    1    0]
 [  86    0    3    0    1    0    5    0    2    3    0]
 [  84    0    0    0    0    3    3    3    3    4    0]
 [  84    0    0    0    1    1    5    1    4    4    0]
 [  82    0    0    2    1    5    2    2    4    2    0]
 [  80    0    3    0    0    2    8    3    1    3    0]
 [  79    0    1    1    0    2    4    3    4    6    0]
 [  74    1    1    2    2    0    5    0    8    5    2]
 [  71    0    3    0    0    3    3    3    4   13    0]
 [  51    0    0    3    0    0    2    2    4    1   37]]
Training time = 2100.8s
Results using also input features for edges
Test accuracy: 0.941
[[6204   26   30   29   25   26   29   23   26   35   47]
 [  11   88    0    0    0    0    1    0    0    0    0]
 [  11    0   87    0    0    1    0    1    0    0    0]
 [  10    1    1   85    0    1    1    1    0    0    0]
 [   9    0    1    1   83    1    3    0    2    0    0]
 [   9    0    0    1    1   83    1    3    0    2    0]
 [   8    0    1    0    2    2   83    0    3    0    1]
 [   8    0    0    1    0    2    2   85    0    2    0]
 [   8    0    0    0    1    0    2    1   86    0    2]
 [   8    0    0    0    0    1    0    1    1   89    0]
 [   8    0    0    0    0    0    2    0    1    1   88]]

"""