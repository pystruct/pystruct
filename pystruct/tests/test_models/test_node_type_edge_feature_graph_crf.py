import pytest
import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal, assert_equal)
from nose.tools import assert_raises

from pystruct.models import NodeTypeEdgeFeatureGraphCRF, EdgeFeatureGraphCRF

from pystruct.inference.linear_programming import lp_general_graph
from pystruct.inference import compute_energy, get_installed
from pystruct.utils import make_grid_edges, edge_list_to_features
from pystruct.datasets import generate_blocks_multinomial



def test_checks():
    g = NodeTypeEdgeFeatureGraphCRF(
                    1                   #how many node type?
                 , [4]                  #how many labels   per node type?
                 , [3]                  #how many features per node type?
                 , np.array([[3]])      #how many features per node type X node type? 
                 )

    g = NodeTypeEdgeFeatureGraphCRF(
                    2                   #how many node type?
                 , [2, 3   ]               #how many labels   per node type?
                 , [4, 5]                  #how many features per node type?
                 , np.array([[1, 2], [2,4]])      #how many features per node type X node type? 
                 )
    
    with pytest.raises(ValueError):
        g = NodeTypeEdgeFeatureGraphCRF(
                        3                   #how many node type?
                     , [2, 3   ]               #how many labels   per node type?
                     , [4, 5]                  #how many features per node type?
                     , np.array([[1, 2], [2,4]])      #how many features per node type X node type? 
                     )
 
    with pytest.raises(ValueError):
        g = NodeTypeEdgeFeatureGraphCRF(
                        2                   #how many node type?
                     , [2, 3   ]               #how many labels   per node type?
                     , [4, 5, 3]                  #how many features per node type?
                     , np.array([[1, 2], [2,4]])      #how many features per node type X node type? 
                     )

    with pytest.raises(ValueError):
        g = NodeTypeEdgeFeatureGraphCRF(
                        3                   #how many node type?
                     , [2, 3   ]               #how many labels   per node type?
                     , [4, 5]                  #how many features per node type?
                     , np.array([[1, 2], [2,4]])      #how many features per node type X node type? 
                     )

    with pytest.raises(ValueError):
        g = NodeTypeEdgeFeatureGraphCRF(
                        2                   #how many node type?
                     , [2, 3   ]               #how many labels   per node type?
                     , [4, 5]                  #how many features per node type?
                     , np.array([[1, 2, 3], [2,3,4]])      #how many features per node type X node type? 
                     )

    with pytest.raises(ValueError):
        g = NodeTypeEdgeFeatureGraphCRF(
                        2                   #how many node type?
                     , [2, 3   ]               #how many labels   per node type?
                     , [4, 5]                  #how many features per node type?
                     , np.array([[1, 2], [99,4]])      #how many features per node type X node type? 
                     )
 
def debug_joint_feature():
    # -------------------------------------------------------------------------------------------
    #print "---MORE COMPLEX GRAPH :) ---------------------------------------------------------------------"
    g = NodeTypeEdgeFeatureGraphCRF(    
                    2                   #how many node type?
                 , [2, 3]               #how many possible labels per node type?
                 , [3, 4]               #how many features per node type?
                 , np.array([  [1, 2]
                             , [2, 3]])      #how many features per node type X node type? 
                    )
 
    l_node_f = [  np.array([ [1,1,1], [2,2,2] ])
                , np.array([ [.11, .12, .13, .14], [.21, .22, .23, .24], [.31, .32, .33, .34]]) 
              ]
    l_edges = [ np.array([[0, 1]]) #type 0 node 0 to type 0 node 0 
              , np.array([[0, 1]])
              , None
              , None
              ]              
    l_edge_f = [  np.array([[.111]])
              , np.array([[.221, .222]])
              , None
              , None
              ]
     
    x = (l_node_f, l_edges, l_edge_f)
    #print "- - - - - - - - - - - - - - - - - - - - - - - - - - - "
    y = np.hstack([   np.array([0, 1]),
                   np.array([0, 1, 2])
            ])
    #print y
    g.initialize(x, y)
    jf = g.joint_feature(x,y)
    #print "joint_feature = \n", `jf`
    #print
    assert_array_equal(jf, jf)
    assert_array_almost_equal(jf
                       , np.array(
      [ 1. , 1., 1.  , 2.,  2., 2.    
       , 0.11 , 0.12 , 0.13 , 0.14 ,   0.21 ,  0.22 ,  0.23 ,  0.24   ,  0.31 ,  0.32 ,  0.33 ,  0.34 
       
       ,  0.   ,  0.111,  0.   ,  0.   ,  0.   ,  0.221,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.222,  0.   ,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.  
       ]))


def get_simple_graph_structure():
    g = NodeTypeEdgeFeatureGraphCRF(
                    1                   #how many node type?
                 , [4]                  #how many labels   per node type?
                 , [3]                  #how many features per node type?
                 , np.array([[3]])      #how many features per node type X node type?                  
                 )
    return g

def get_simple_graph():
    node_f = [ np.array([[1,1,1], 
                         [2,2,2]]) 
              ]
    edges  = [ np.array([[0,1]]) 
              ]    #an edge from 0 to 1
    edge_f = [ np.array([[3,3,3]]) 
              ]
    return (node_f, edges, edge_f)

def get_simple_graph2():
    node_f = [ np.array([  [1,1,1]
                         , [2,2,2]]) ]
    edges  = [ np.array( [[0,1], #an edge from 0 to 1
                          [0,0]  #an edge from 0 to 0
                          ])  ]  
    edge_f = [ np.array([
        [3,3,3],
        [4,4,4]
             ]) ]
    return (node_f, edges, edge_f)

def test_flatten_unflattenY():
    
    g, (node_f, edges, edge_f) = get_simple_graph_structure(), get_simple_graph()
    y = np.array([1,2])
    l_nf = [ np.zeros((2,3)) ]  #list of node feature , per type
    X = (l_nf, None, None)      #we give no edge
    y_ref = [ np.array([1,2]) ]
    assert all( [ (y_typ1 == y_typ2).all() for y_typ1, y_typ2 in zip(g.unflattenY(X, y), y_ref) ])
    
    assert (y == g.flattenY(g.unflattenY(X, y))).all()

    #============================================    
    g, x, y = more_complex_graph()

    Y = [ np.array([0, 0]) 
        , np.array([0, 0, 0])   #we start again at zero on 2nd type
        ]
    
    y = np.hstack([ np.array([0, 0]) 
                  , 2+np.array([0, 0, 0])
         ])
    l_nf = [  np.zeros( (2,3) ), np.zeros( (3, 4) )]  #2 node with 3 features, 3 node with 4 features
    X = (l_nf, None, None)      #we give no edge
    assert (g.flattenY(Y) == y).all()
    #print g.unflattenY(X, y)
    assert all( [ (y_typ1 == y_typ2).all() for y_typ1, y_typ2 in zip(g.unflattenY(X, y), Y) ])
    
    l_nf = [  np.zeros( (1,3) ), np.zeros( (3, 4) )]  #2 node with 3 features, 3 node with 4 features
    X = (l_nf, None, None)      #we give no edge
    assert_raises(ValueError, g.unflattenY, X, y)
        
def test_joint_feature():
 
    #print "---SIMPLE---------------------------------------------------------------------"
    g, (node_f, edges, edge_f) = get_simple_graph_structure(), get_simple_graph()
     
    x = (node_f, edges, edge_f)
    #print "- - - - - - - - - - - - - - - - - - - - - - - - - - - "
    y = np.array([1,2])
    
#     y = np.array([1,0])
    #print y
    g.initialize(x, y)
    jf = g.joint_feature(x,y)
    #print "joint_feature = \n", `jf`
    #print
    assert_array_equal(g.joint_feature(x,y)
                       , np.array([ 0.,  0.,  0.,  1.,  1.,  1.,  2.,  2.,  2.,  0.,  0.,  0.
                                   ,  0.,
        0.,  0.,  0.,  0.,  0.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  3.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  3.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
                       )
    
    #print "- - - - - - - - - - - - - - - - - - - - - - - - - - - "
    y = np.array([0,0])
    #print y
    g.initialize(x, y)
    jf = g.joint_feature(x,y)
    #print "joint_feature = \n", `jf`
    #print
    assert_array_equal(g.joint_feature(x,y)
                       , np.array([ 3.,  3.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  3.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
                       )

    #print "- - - - - - - - - - - - - - - - - - - - - - - - - - - "
    y = np.array([0,1])
    node_f = [ np.array([[1.1,1.2,1.3], [2.1,2.2,2.3]]) ]
    edge_f = [ np.array([[3.1,3.2,3.3]]) ]
    x = (node_f, edges, edge_f)
    g.initialize(x, y)
    jf = g.joint_feature(x,y)
    #print "joint_feature = \n", `jf`
    
    assert_array_equal(g.joint_feature(x,y)
                       , np.array([ 1.1,  1.2,  1.3,  2.1,  2.2,  2.3,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  3.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  3.2,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  3.3,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ])
                       )
    #print "---SIMPLE + 2nd EDGE--------------------------------------------------------"
    node_f, edges, edge_f = get_simple_graph2()

    x = (node_f, edges, edge_f)
    #print "- - - - - - - - - - - - - - - - - - - - - - - - - - - "
    y = np.array([1,2])
    #print y
    g.initialize(x, y)
    jf = g.joint_feature(x,y)
    #print "joint_feature = \n", `jf`
    #print
    assert_array_equal(jf
                       , np.array([ 0.,  0.,  0.,  1.,  1.,  1.,  2.,  2.,  2.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  4.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  4.,  3.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  4.,  3.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
                       )
    #print "- - - - - - - - - - - - - - - - - - - - - - - - - - - "
    y = np.array([0,0])
    #print y
    g.initialize(x, y)
    #print "joint_feature = \n", `g.joint_feature(x,y)`
    #print
    assert_array_equal(g.joint_feature(x,y)
                       , np.array([ 3.,  3.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  7.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  7.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  7.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
                       )

def more_complex_graph():
    g = NodeTypeEdgeFeatureGraphCRF(    
                    2                   #how many node type?
                 , [2, 3]               #how many labels   per node type?
                 , [3, 4]               #how many features per node type?
                 , np.array([  [1, 2]
                             , [2, 3]])      #how many features per node type X node type? 
                 )
 
#     nodes  = np.array( [[0,0], [0,1], [1, 0], [1, 1], [1, 2]] )  
    node_f = [  np.array([ [1,1,1], [2,2,2] ])
              , np.array([ [.11, .12, .13, .14], [.21, .22, .23, .24], [.31, .32, .33, .34]]) 
              ]
    edges  = [ np.array( [  [0,1]   #an edge from 0 to 1
                        ])
              , np.array( [ 
                          [0,0]   #an edge from typ0:0 to typ1:0 
                        ])
              , None
              , None
              ]    
    edge_f = [ np.array([[.111]])
              , np.array([[.221, .222]])
              , None
              , None
              ]
     
    x = (node_f, edges, edge_f)
    y = np.hstack([ np.array([0, 0]) 
                  , 2+np.array([0, 0, 0])
         ])
    return g, x, y
        
def test_joint_feature2():

    # -------------------------------------------------------------------------------------------
    #print "---MORE COMPLEX GRAPH :) ---------------------------------------------------------------------"
    g, x, y = more_complex_graph()
    #print y
    
    
    g.initialize(x, y)
    jf = g.joint_feature(x,y)
    #print "joint_feature = \n", `jf`
    #print
    assert_array_equal(jf, jf)
    assert_array_almost_equal(jf
                       , np.array([ 3.   ,  3.   ,  3.   ,  0.   ,  0.   ,  0.   ,  0.63 ,  0.66 ,
        0.69 ,  0.72 ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
        0.   ,  0.   ,  0.111,  0.   ,  0.   ,  0.   ,  0.221,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.222,  0.   ,  0.   ,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ]))

    #print "---MORE COMPLEX GRAPH :) -- BIS -------------------------------------------------------------------"
    g = NodeTypeEdgeFeatureGraphCRF(    
                    2                   #how many node type?
                 , [2, 3]               #how many labels   per node type?
                 , [3, 4]               #how many features per node type?
                 , np.array([  [1, 2]
                             , [2, 3]])      #how many features per node type X node type? 
                 )
 
    node_f = [  np.array([ [1,1,1], [2,2,2] ])
              , np.array([ [.11, .12, .13, .14], [.21, .22, .23, .24], [.31, .32, .33, .34]]) 
              ]
    edges  = [ np.array( [  [0,1]] ),   #an edge from 0 to 1
               np.array( [  [0,2]] )   #an edge from 0 to 2
               , None, None
                        ]    
    edge_f = [ np.array([[.111]])
              , np.array([[.221, .222]])
              , None
              , None
              ]
     
    x = ( node_f, edges, edge_f)
    #print "- - - - - - - - - - - - - - - - - - - - - - - - - - - "
    y = np.hstack([np.array([0, 1]),
                   2+np.array([0, 1, 2])])
    #print y
    g.initialize(x, y)
    jf = g.joint_feature(x,y)
    #print "joint_feature = \n", `jf`
    #print
    assert_array_equal(jf, jf)
    assert_array_almost_equal(jf
                       , np.array([ 1.   ,  1.   ,  1.   ,  2.   ,  2.   ,  2.   ,  0.11 ,  0.12 ,
        0.13 ,  0.14 ,  0.21 ,  0.22 ,  0.23 ,  0.24 ,  0.31 ,  0.32 ,
        0.33 ,  0.34 ,  0.   ,  0.111,  0.   ,  0.   ,  0.   ,  0.   ,
        0.221,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.222,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ]))
    
    #print "MORE COMPLEX GRAPH :) -- BIS  OK"
    #print "--- REORDERED MORE COMPLEX GRAPH :) ---------------------------------------------------------------------"
    node_f = [  np.array([ [2,2,2], [1,1,1] ])
              , np.array([ [.31, .32, .33, .34], [.11, .12, .13, .14], [.21, .22, .23, .24]]) 
              ]
    edges  = [ np.array( [  [1, 0]] ),
               np.array( [  [1,0]] )   #an edge from 0 to 2
               , None, None
                        ]    
    edge_f = [ np.array([[.111]])
              , np.array([[.221, .222]])
              , None
              , None
              ]
     
    x = ( node_f, edges, edge_f)
    #print "- - - - - - - - - - - - - - - - - - - - - - - - - - - "
    y = np.hstack([np.array([1, 0]),
                   2+np.array([2, 0, 1])])
    #print y
    g.initialize(x, y)
    jf = g.joint_feature(x,y)
    #print "joint_feature = \n", `jf`
    #print
    assert_array_equal(jf, jf)
    assert_array_almost_equal(jf
                       , np.array([ 1.   ,  1.   ,  1.   ,  2.   ,  2.   ,  2.   ,  0.11 ,  0.12 ,
        0.13 ,  0.14 ,  0.21 ,  0.22 ,  0.23 ,  0.24 ,  0.31 ,  0.32 ,
        0.33 ,  0.34 ,  0.   ,  0.111,  0.   ,  0.   ,  0.   ,  0.   ,
        0.221,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.222,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ]))
    
def test_joint_feature3():

    # -------------------------------------------------------------------------------------------
    #print "---MORE COMPLEX GRAPH AGAIN :) ---------------------------------------------------------------------"
    g = NodeTypeEdgeFeatureGraphCRF(    
                    2                   #how many node type?
                 , [2, 3]               #how many labels   per node type?
                 , [3, 4]               #how many features per node type?
                 , np.array([  [0, 2]
                             , [2, 3]])      #how many features per node type X node type? 
                 )
 
#     nodes  = np.array( [[0,0], [0,1], [1, 0], [1, 1], [1, 2]] )  
    node_f = [  np.array([ [1,1,1], [2,2,2] ])
              , np.array([ [.11, .12, .13, .14], [.21, .22, .23, .24], [.31, .32, .33, .34]]) 
              ]
    edges  = [ None
              , np.array( [ 
                          [0,1]   #an edge from typ0:0 to typ1:1 
                        ])
              , None
              , np.array( [ 
                          [0,1],   #an edge from typ0:0 to typ1:1 
                          [1,2]   #an edge from typ1:1 to typ1:2 
                        ])
              ]    
    edge_f = [ None
              , np.array([[.221, .222]])
              , None
              , np.array([[.01,  .02,  .03 ],
                          [.001, .002, .003]])
              ]
     
    x = (node_f, edges, edge_f)
    #print "- - - - - - - - - - - - - - - - - - - - - - - - - - - "
    y = np.hstack([ np.array([0, 0]) 
                  , 2+np.array([0, 0, 0])
         ])
    #print y
    g.initialize(x, y)
    #print g.size_unaries
    #print g.size_pairwise
    jf = g.joint_feature(x,y)
    #print "joint_feature = \n", `jf`
    #print
    assert_array_equal(jf, jf)
    assert_array_almost_equal(jf
                       , np.array([ 3.   ,  3.   ,  3.   ,  0.   ,  0.   ,  0.   ,  
                                   0.63 ,  0.66 ,  0.69 ,  0.72 ,    0. , 0.,  0.,  0.   ,    0.,  0., 0. ,  0.,
                                   #edges 0 to 0 2x2 states
                                   #typ0 typ0 EMPTY
                                   #typ0 typ1
                                   .221, 0., 0.,   0., 0., 0.,
                                   .222, 0., 0.,   0., 0., 0.,
                                   #typ1 typ0
                                    0., 0., 0.,   0., 0., 0.,
                                    0., 0., 0.,   0., 0., 0.,
                                   #typ1 typ1
                                    0.011, 0., 0.,   0., 0., 0.,   0., 0., 0., 
                                    0.022, 0., 0.,   0., 0., 0.,   0., 0., 0.,
                                    0.033, 0., 0.,   0., 0., 0.,   0., 0., 0.
                                   ])
                              )

    #print "- - - - - - - - - - - - - - - - - - - - - - - - - - - "
    y = np.hstack([ np.array([0, 1]) 
                  , 2+np.array([1, 1, 0])
         ])
    #print y
    g.initialize(x, y)
    jf = g.joint_feature(x,y)
    #print "joint_feature = \n", `jf`
    #print
    assert_array_equal(jf, jf)
    assert_array_almost_equal(jf
                       , np.array([ 1.   ,  1.   ,  1.   ,  2.   ,  2.   ,  2.   ,  
                                   .31, .32, .33, .34 ,    .32, .34, .36, .38   ,    0.,  0., 0. ,  0.,
                                   #edges 0 to 0 2x2 states
                                   #typ0 typ0 EMPTY
                                   #typ0 typ1
                                   0., .221, 0.,   0., 0., 0.,
                                   0., .222, 0.,   0., 0., 0.,
                                   #typ1 typ0
                                    0., 0., 0.,   0., 0., 0.,
                                    0., 0., 0.,   0., 0., 0.,
                                   #typ1 typ1
                                    0., 0., 0.,   0.001, 0.01, 0.,   0., 0., 0., 
                                    0., 0., 0.,   0.002, 0.02, 0.,   0., 0., 0.,
                                    0., 0., 0.,   0.003, 0.03, 0.,   0., 0., 0.
                                   ])
                              )

    w = np.array([ 1,1,1, 2,2,2,   10,10,10,10, 20,20,20,20, 30,30,30,30 ]
                  +[1.0]*51, dtype=np.float64
                  )
    #print `w`
    ret_u = g._get_unary_potentials(x, w)
    #print `ret_u`
    assert len(ret_u) == 2
    assert_array_almost_equal(ret_u[0], np.array([   #n_nodes x n_states
                                              [3,  6],
                                              [6, 12]]))

    assert_array_almost_equal(ret_u[1], np.array([   #n_nodes x n_states
                                                [5, 10, 15],
                                                [9, 18, 27],
                                                [13, 26, 39]]))
    
    assert len(w) == g.size_joint_feature
    ret_pw = g._get_pairwise_potentials(x, w)
    #     for _pw in ret_pw:
    #         print "_pw ", `_pw`
    pw00, pw01, pw10, pw11 = ret_pw
    assert len(pw00) == 0
    assert_array_almost_equal(pw01,np.array([   #n_edges, n_states, n_states
                                  [[0.443,  0.443,  0.443],
                                   [0.443,  0.443,  0.443]]
                                               ]))
    assert len(pw10) == 0
    
    assert_array_almost_equal(pw11,np.array([   #n_edges, n_states, n_states
                                  [[0.06 ,  0.06 ,  0.06],
                                   [0.06 ,  0.06 ,  0.06],
                                   [0.06 ,  0.06 ,  0.06]]
                                             ,
                                  [[0.006,  0.006,  0.006],
                                   [0.006,  0.006,  0.006],
                                   [0.006,  0.006,  0.006]]        
                                               ]))



def test_unary_potentials():
    #print "---SIMPLE---------------------------------------------------------------------"
    #g, (node_f, edges, edge_f) = get_simple_graph_structure(), get_simple_graph()

    g = NodeTypeEdgeFeatureGraphCRF(
                    1                   #how many node type?
                 , [4]                  #how many labels   per node type?
                 , [3]                  #how many features per node type?
                 , np.array([[3]])      #how many features per node type X node type?                  
                 )
    node_f = [ np.array([[1,1,1], 
                         [2,2,2]]) 
              ]
    edges  = [ np.array([[0,1]]) 
              ]    #an edge from 0 to 1
    edge_f = [ np.array([[3,3,3]]) 
              ]
    x = (node_f, edges, edge_f)
    #print "- - - - - - - - - - - - - - - - - - - - - - - - - - - "
    y = np.hstack([ np.array([1,2])])
#     y = np.array([1,0])
    #print y
    g.initialize(x, y)
    
    gref = EdgeFeatureGraphCRF(4,3,3)
    xref = (node_f[0], edges[0], edge_f[0])
    wref = np.arange(gref.size_joint_feature)
    potref = gref._get_unary_potentials(xref, wref)
    #print `potref`
    
    w = np.arange(g.size_joint_feature)
    pot = g._get_unary_potentials(x, w)
    #print `pot`
    assert_array_equal(pot, [potref])

    pwpotref = gref._get_pairwise_potentials(xref, wref)
    #print `pwpotref`
    pwpot = g._get_pairwise_potentials(x, w)
    #print `pwpot`
    assert_array_equal(pwpot, [pwpotref])
 
# def test_inference_util():
#     g = NodeTypeEdgeFeatureGraphCRF(    
#                     3                   #how many node type?
#                  , [2, 3, 1]               #how many labels   per node type?
#                  , [3, 4, 1]               #how many features per node type?
#                  , np.array([  [1, 2, 2]
#                              , [2, 3, 2]
#                              , [2, 2, 1]])      #how many features per node type X node type? 
#                  )
#     node_f = [  np.array([ [2,2,2], [1,1,1] ])
#               , np.array([ [.31, .32, .33, .34], [.11, .12, .13, .14], [.21, .22, .23, .24]]) 
#               , np.array([ [77], [88], [99]]) 
#               ]
#     edges  = [ np.array( [  [1, 0]] ),
#                np.array( [  [1,0]] )   #an edge from 0 to 2
#                , None
#                
#                , None
#                , None
#                , None
#                
#                , np.array( [[1,1]] )
#                , None
#                , None                        ]    
# 
#     x = ( node_f, edges, None)
#     
#     reindexed_exdges = g._index_all_edges(x)
#     #print `reindexed_exdges`
#     assert_array_equal(reindexed_exdges,
#                        np.array( [[1,0],
#                                   [1,2],
#                                   [6,1]]))
#     

# def report_model_config(crf):
#     print crf.n_states
#     print crf.n_features
#     print crf.n_edge_features

def inference_data():
    """
    Testing with a single type of nodes. Must do as well as EdgeFeatureGraphCRF
    """
    # Test inference with different weights in different directions

    X, Y = generate_blocks_multinomial(noise=2, n_samples=1, seed=1)
    x, y = X[0], Y[0]
    n_states = x.shape[-1]

    edge_list = make_grid_edges(x, 4, return_lists=True)
    edges = np.vstack(edge_list)

    pw_horz = -1 * np.eye(n_states)
    xx, yy = np.indices(pw_horz.shape)
    # linear ordering constraint horizontally
    pw_horz[xx > yy] = 1

    # high cost for unequal labels vertically
    pw_vert = -1 * np.eye(n_states)
    pw_vert[xx != yy] = 1
    pw_vert *= 10

    # generate edge weights
    edge_weights_horizontal = np.repeat(pw_horz[np.newaxis, :, :],
                                        edge_list[0].shape[0], axis=0)
    edge_weights_vertical = np.repeat(pw_vert[np.newaxis, :, :],
                                      edge_list[1].shape[0], axis=0)
    edge_weights = np.vstack([edge_weights_horizontal, edge_weights_vertical])

    # do inference
    res = lp_general_graph(-x.reshape(-1, n_states), edges, edge_weights)

    edge_features = edge_list_to_features(edge_list)
    x = ([x.reshape(-1, n_states)], [edges], [edge_features])
    y = y.ravel()
    return x, y, pw_horz, pw_vert, res, n_states
    
def test_inference_ad3plus():
    
    x, y, pw_horz, pw_vert, res, n_states = inference_data()
    # same inference through CRF inferface
    crf = NodeTypeEdgeFeatureGraphCRF(1, [3], [3], [[2]]
                                      , inference_method="ad3+")   
    crf.initialize(x, y)
    #crf.initialize([x], [y])
    w = np.hstack([np.eye(3).ravel(), -pw_horz.ravel(), -pw_vert.ravel()])
    y_pred = crf.inference(x, w, relaxed=True)
    if isinstance(y_pred, tuple):
        # ad3 produces an integer result if it found the exact solution
        #np.set_printoptions(precision=2, threshold=9999)
        assert_array_almost_equal(res[0], y_pred[0][0].reshape(-1, n_states), 5)
        assert_array_almost_equal(res[1], y_pred[1][0], 5)
        assert_array_equal(y, np.argmax(y_pred[0][0], axis=-1), 5)

        # again, this time discrete predictions only
    crf = NodeTypeEdgeFeatureGraphCRF(1, [3], [3], [[2]]
                                      , inference_method="ad3+")   
    #crf.initialize([x], [y])
    w = np.hstack([np.eye(3).ravel(), -pw_horz.ravel(), -pw_vert.ravel()])
    crf.initialize(x)
    y_pred = crf.inference(x, w, relaxed=False)
    assert_array_equal(y, y_pred)

def test_inference_ad3():
    
    x, y, pw_horz, pw_vert, res, n_states = inference_data()
    # same inference through CRF inferface
    crf = NodeTypeEdgeFeatureGraphCRF(1, [3], [3], [[2]]
                                      , inference_method="ad3")   
    crf.initialize(x, y)
    #crf.initialize([x], [y])
    w = np.hstack([np.eye(3).ravel(), -pw_horz.ravel(), -pw_vert.ravel()])
    y_pred = crf.inference(x, w, relaxed=True)
    if isinstance(y_pred, tuple):
        # ad3 produces an integer result if it found the exact solution
        #np.set_printoptions(precision=2, threshold=9999)
        assert_array_almost_equal(res[0], y_pred[0][0].reshape(-1, n_states), 5)
        assert_array_almost_equal(res[1], y_pred[1][0], 5)
        assert_array_equal(y, np.argmax(y_pred[0][0], axis=-1), 5)

        # again, this time discrete predictions only
    crf = NodeTypeEdgeFeatureGraphCRF(1, [3], [3], [[2]]
                                      , inference_method="ad3")   
    #crf.initialize([x], [y])
    w = np.hstack([np.eye(3).ravel(), -pw_horz.ravel(), -pw_vert.ravel()])
    crf.initialize(x)
    y_pred = crf.inference(x, w, relaxed=False)
    assert_array_equal(y, y_pred)

def test_joint_feature_discrete():
    """
    Testing with a single type of nodes. Must de aw well as EdgeFeatureGraphCRF
    """
    X, Y = generate_blocks_multinomial(noise=2, n_samples=1, seed=1)
    x, y = X[0], Y[0]
    edge_list = make_grid_edges(x, 4, return_lists=True)
    edges = np.vstack(edge_list)
    edge_features = edge_list_to_features(edge_list)
    x = ([x.reshape(-1, 3)], [edges], [edge_features])
    y_flat = y.ravel()
    #for inference_method in get_installed(["lp", "ad3", "qpbo"]):
    if True:
        crf = NodeTypeEdgeFeatureGraphCRF(1, [3], [3], [[2]])
        joint_feature_y = crf.joint_feature(x, y_flat)
        assert_equal(joint_feature_y.shape, (crf.size_joint_feature,))
        # first horizontal, then vertical
        # we trust the unaries ;)
        n_states = crf.l_n_states[0]
        n_features = crf.l_n_features[0]
        pw_joint_feature_horz, pw_joint_feature_vert = joint_feature_y[n_states *
                                         n_features:].reshape(
                                             2, n_states, n_states)
        assert_array_equal(pw_joint_feature_vert, np.diag([9 * 4, 9 * 4, 9 * 4]))
        vert_joint_feature = np.diag([10 * 3, 10 * 3, 10 * 3])
        vert_joint_feature[0, 1] = 10
        vert_joint_feature[1, 2] = 10
        assert_array_equal(pw_joint_feature_horz, vert_joint_feature)

def test_joint_feature_continuous():
    """
    Testing with a single type of nodes. Must de aw well as EdgeFeatureGraphCRF
    """
    # FIXME
    # first make perfect prediction, including pairwise part
    X, Y = generate_blocks_multinomial(noise=2, n_samples=1, seed=1)
    x, y = X[0], Y[0]
    n_states = x.shape[-1]
    edge_list = make_grid_edges(x, 4, return_lists=True)
    edges = np.vstack(edge_list)
    edge_features = edge_list_to_features(edge_list)
    #x = (x.reshape(-1, 3), edges, edge_features)
    x = ([x.reshape(-1, 3)], [edges], [edge_features])
    y = y.ravel()

    pw_horz = -1 * np.eye(n_states)
    xx, yy = np.indices(pw_horz.shape)
    # linear ordering constraint horizontally
    pw_horz[xx > yy] = 1

    # high cost for unequal labels vertically
    pw_vert = -1 * np.eye(n_states)
    pw_vert[xx != yy] = 1
    pw_vert *= 10

    # create crf, assemble weight, make prediction
#     for inference_method in get_installed(["lp", "ad3"]):
#         crf = EdgeFeatureGraphCRF(inference_method=inference_method)
    if True:
        crf = NodeTypeEdgeFeatureGraphCRF(1, [3], [3], [[2]])
        
        w = np.hstack([np.eye(3).ravel(), -pw_horz.ravel(), -pw_vert.ravel()])
        #crf.initialize([x], [y])
        #report_model_config(crf)
        crf.initialize(x, y)
        
        y_pred = crf.inference(x, w, relaxed=True)

        # compute joint_feature for prediction
        joint_feature_y = crf.joint_feature(x, y_pred)
        assert_equal(joint_feature_y.shape, (crf.size_joint_feature,))
        # FIXME
        # first horizontal, then vertical
        # we trust the unaries ;)
        #pw_joint_feature_horz, pw_joint_feature_vert = joint_feature_y[crf.n_states *
                                 #crf.n_features:].reshape(2,
                                                          #crf.n_states,
                                                          #crf.n_states)

def test_energy_continuous():
    # make sure that energy as computed by ssvm is the same as by lp
    np.random.seed(0)
    #for inference_method in get_installed(["lp", "ad3"]):
    if True:
        found_fractional = False
        crf = NodeTypeEdgeFeatureGraphCRF(1, [3], [3], [[2]])

        while not found_fractional:
            x = np.random.normal(size=(7, 8, 3))
            edge_list = make_grid_edges(x, 4, return_lists=True)
            edges = np.vstack(edge_list)
            edge_features = edge_list_to_features(edge_list)
            x = ([x.reshape(-1, 3)], [edges], [edge_features])

            unary_params = np.random.normal(size=(3, 3))
            pw1 = np.random.normal(size=(3, 3))
            pw2 = np.random.normal(size=(3, 3))
            w = np.hstack([unary_params.ravel(), pw1.ravel(), pw2.ravel()])
            crf.initialize(x)
            res, energy = crf.inference(x, w, relaxed=True, return_energy=True)
            found_fractional = np.any(np.max(res[0], axis=-1) != 1)
            joint_feature = crf.joint_feature(x, res)
            energy_svm = np.dot(joint_feature, w)

            assert_almost_equal(energy, -energy_svm)

def test_energy_discrete():
#     for inference_method in get_installed(["qpbo", "ad3"]):
#         crf = EdgeFeatureGraphCRF(n_states=3,
#                                   inference_method=inference_method,
#                                   n_edge_features=2, n_features=3)
        crf = NodeTypeEdgeFeatureGraphCRF(1, [3], [3], [[2]])
        
        for i in range(10):
            x = np.random.normal(size=(7, 8, 3))
            edge_list = make_grid_edges(x, 4, return_lists=True)
            edges = np.vstack(edge_list)
            edge_features = edge_list_to_features(edge_list)
            x = ([x.reshape(-1, 3)], [edges], [edge_features])

            unary_params = np.random.normal(size=(3, 3))
            pw1 = np.random.normal(size=(3, 3))
            pw2 = np.random.normal(size=(3, 3))
            w = np.hstack([unary_params.ravel(), pw1.ravel(), pw2.ravel()])
            crf.initialize(x)
            y_hat = crf.inference(x, w, relaxed=False)
            #flat_edges = crf._index_all_edges(x)
            energy = compute_energy(crf._get_unary_potentials(x, w)[0],
                                    crf._get_pairwise_potentials(x, w)[0], edges, #CAUTION: pass the flatened edges!!
                                    y_hat)

            joint_feature = crf.joint_feature(x, y_hat)
            energy_svm = np.dot(joint_feature, w)

            assert_almost_equal(energy, energy_svm)


if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=9999)

    if 0: 
        debug_joint_feature()
    if 1:
        test_flatten_unflattenY()
        
    if 1:
        test_joint_feature()
    if 1:
        test_joint_feature2()
    if 1:
        test_joint_feature3()
        
    if 1: test_unary_potentials()
#     if 1: test_inference_util()
    if 1: test_inference_ad3()
    if 1: test_inference_ad3plus()
    if 1: test_joint_feature_discrete()
    if 1: test_joint_feature_continuous()
    if 1: test_energy_continuous()
    if 1: test_energy_discrete()
    
    #print "OK"
