"""
==============================================
Conditional Interactions on the Snakes Dataset
==============================================

This is a variant of plot_snakes.py 

Snake are hidding, so another task is both to determine if a snake is in the picture, and
identify its head to tail body.  

We use the Logit and some picture feature to categorize pictures (only this task)


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
import time

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

from pystruct.datasets import load_snakes

from plot_snakes import  one_hot_colors, neighborhood_feature, prepare_data

from plot_hidden_snakes import shufflePictureCells, shuffleSnakeCells, changeOneSnakeCell, augmentWithNoSnakeImages, shuffle_in_unison
from plot_hidden_snakes_typed import prepare_picture_data

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


if __name__ == '__main__':
    print("Please be patient. Learning will take 5-20 minutes.")
    snakes = load_snakes()
    X_train, Y_train = snakes['X_train'], snakes['Y_train']
    
    bADD_HIDDEN_SNAKES = True
    #bADD_HIDDEN_SNAKES = False
    #JL
    #X_train, Y_train = X_train[:10], Y_train[:10]
    print len(X_train), len(Y_train)
    #print `X_train[0]`

    if bADD_HIDDEN_SNAKES:
        nb_hidden, X_train, Y_train = augmentWithNoSnakeImages(X_train, Y_train, "train", False)
        print len(X_train), len(Y_train)
        Y_train_pict = np.array([1]*(len(X_train)-nb_hidden) + [0]*nb_hidden)
    
    if False:
        #show the faked pictures
        for ix, x in enumerate(X_train): plot_snake(shufflePictureCells(x))  

    X_train = [one_hot_colors(x) for x in X_train]

    X_train, Y_train, Y_train_pict = shuffle_in_unison(X_train, Y_train, Y_train_pict)

    X_train_pict_feat = prepare_picture_data(X_train)
    X_train_pict_feat = np.vstack(X_train_pict_feat)
    print "X_train_pict_feat.shape ", X_train_pict_feat.shape
    lr = LogisticRegression(class_weight='balanced')
    dicGS = {'C':[0.1, 0.5, 1.0, 2.0] }
    dicGS = {'C':[1.0] }
    mdl = GridSearchCV(lr , dicGS)   
    
    print "-training a logistic regression model on pictures"
    mdl.fit(X_train_pict_feat, Y_train_pict)
    
    # --- TEST
    X_test, Y_test = snakes['X_test'], snakes['Y_test']
    print "TEST len=", len(X_test)
    if bADD_HIDDEN_SNAKES:
        nb_hidden, X_test, Y_test = augmentWithNoSnakeImages(X_test, Y_test, "test", bOneHot=False)
        print "TEST len=", len(X_test)
        Y_test_pict = np.array([1]*(len(X_test)-nb_hidden) + [0]*nb_hidden)
    
    X_test = [one_hot_colors(x) for x in X_test]
    X_test_pict_feat = prepare_picture_data(X_test)
    X_test_pict_feat = np.vstack(X_test_pict_feat)

    Y_pred = mdl.predict( X_test_pict_feat )
    print("Results using only directional features for edges")
    print("Test accuracy: %.3f"
          % accuracy_score(Y_test_pict, Y_pred))
    print(confusion_matrix(Y_test_pict, Y_pred))    
    
    
"""
 
 
 """