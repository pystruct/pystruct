# -*- coding: utf-8 -*-

"""
    Pairwise CRF with features/strength associated to each edge and different types of nodes

    Copyright Xerox(C) 2017 JL. Meunier

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import numpy as np

from .typed_crf import TypedCRF

class NodeTypeEdgeFeatureGraphCRF(TypedCRF):
    """
    Pairwise CRF with features/strength associated to each edge and different types of nodes

    Pairwise potentials are asymmetric and shared over all edges of same type.
    They are weighted by an edge-specific features, though.
    This allows for contrast sensitive potentials or directional potentials
    (using a {-1, +1} encoding of the direction for example).

    More complicated interactions are also possible, of course.


    Parameters
    ----------
    n_types : number of node types
    
    l_n_states : list of int, default=None
        Number of states per type of variables. 

    l_n_features : list of int, default=None
        Number of features per type of node. 

    a_n_edge_features: an array of shape (n_types, n_types) given the number of features as a function of the node types
    
    class_weight : None, or list of array-like
        Class weights. If a list of array-like is passed, the Ith one must have length equal to l_n_states[i]
        None means equal class weights (across node types)


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

    def __init__(self
                 , n_types                  #how many node type?
                 , l_n_states               #how many labels   per node type?
                 , l_n_features             #how many features per node type?
                 , a_n_edge_features        #how many features per edge type?
                 , inference_method="ad3" 
                 , l_class_weight=None):    #class_weight      per node type or None           <list of array-like> or None
        
        #internal stuff
        #how many features per node type X node type?      <array-like> (MUST be symmetric!)
        self.a_n_edge_features = np.array(a_n_edge_features)
        if self.a_n_edge_features.shape   != (n_types, n_types):
            raise ValueError("Expected a feature number matrix for edges of shape (%d, %d), got %s."%(n_types, n_types, self.a_n_edge_features.shape))
        self.a_n_edge_features = self.a_n_edge_features.reshape(n_types, n_types)
        if not (self.a_n_edge_features == self.a_n_edge_features.T).all():
            raise ValueError("Expected a symmetric array of edge feature numbers")
        
        self._n_edge_features  = self.a_n_edge_features.sum(axis=None)   #total number of (edge) features

        TypedCRF.__init__(self, n_types, l_n_states, l_n_features, inference_method=inference_method, l_class_weight=l_class_weight)
        
        self._get_pairwise_potentials_initialize()

    def _set_size_joint_feature(self):
        """
        We have:
        - 1 weight per node feature per label per node type
        - 1 weight per edge feature per label of node1 type, per label of node2 type
        """
        if self.l_n_features:
            self.size_unaries = sum(  n_states * n_features for n_states, n_features in zip(self.l_n_states, self.l_n_features) )
        
            self.size_pairwise = 0  #detailed non-optimized computation to make things clear
            for typ1,typ2 in self._iter_type_pairs():
                self.size_pairwise += self.a_n_edge_features[typ1,typ2] * self.l_n_states[typ1] * self.l_n_states[typ2]
                #print "\t %d = %d x %d x %d"%(self.a_n_edge_features[typ1,typ2] * self.l_n_states[typ1] * self.l_n_states[typ2], self.a_n_edge_features[typ1,typ2] , self.l_n_states[typ1] , self.l_n_states[typ2])
            self.size_joint_feature = self.size_unaries + self.size_pairwise
        
            #print "size = ",  self.size_unaries, " + " , self.size_pairwise

    def __repr__(self):
        return ("%s(n_states: %d, inference_method: %s, n_features: %d, "
                "n_edge_features: %d)"
                % (type(self).__name__, self.l_n_states, self.inference_method,
                   self.l_n_features, self.a_n_edge_features))

    def _check_size_x(self, x):
        l_edges = self._get_edges(x)
        if len(l_edges) != self.n_types**2:
            raise ValueError("Expected %d edge arrays or None"%(self.n_types**2))
        l_edge_features = self._get_edge_features(x) 
        if len(l_edge_features) != self.n_types**2:
            raise ValueError("Expected %d edge feature arrays or None"%(self.n_types**2))

        TypedCRF._check_size_x(self, x)
        
        #check that we have in total 1 feature vector per edge
        for edges, edge_features in zip(l_edges, l_edge_features):
            if edges is None or edge_features is None: 
                if edges is None and edge_features is None: continue
                if edges is None:
                    raise ValueError("Empty edge array but non empty edge-feature array, for same type of edge")
                else:
                    raise ValueError("Empty edge-feature array but non empty edge array, for same type of edge")
            if edge_features.ndim != 2:
                raise ValueError("Expected a 2 dimensions edge feature arrays")
            if len(edges) != len(edge_features):
                raise ValueError("Edge and edge feature matrices must have same size in 1st dimension")

        #check edge feature size 
        for typ1,typ2 in self._iter_type_pairs():
            edge_features = self._get_edge_features_by_type(x, typ1, typ2) 
            if edge_features is None: continue
            if edge_features.shape[1] != self.a_n_edge_features[typ1,typ2]:
                raise ValueError("Types %d x %d: bad number of edge features. expected %d got %d"%(typ1,typ2, self.a_n_edge_features[typ1,typ2], edge_features.shape[1]))

    def _get_edge_features(self, x, bClean=False):
        if bClean:
            return [ np.empty((0,0)) if o is None or len(o)==0 else o for o in x[2]]
        else:
            return x[2]
    def _get_edge_features_by_type(self, x, typ1, typ2):
        return x[2][typ1*self.n_types+typ2] 

    def _get_pairwise_potentials_initialize(self):
        """
        Putting in cache the params required to build the pairwise potentials given x and w
        """
        self._cache_pairwise_potentials = list()
#         i_w, n_states1, n_states2, i_states1, i_states2 = 0, 0, 0, 0, 0
#         for (typ1, typ2) in self._iter_type_pairs():

        i_w, n_states1, i_states1 = 0, 0, 0

        for typ1 in xrange(self.n_types):
            n_states1               = self.l_n_states[typ1]
            i_states1_stop          = i_states1 + n_states1
            n_states2, i_states2 = 0, 0
            for typ2 in xrange(self.n_types):
                n_features              = self.a_n_edge_features[typ1, typ2]
                n_states2               = self.l_n_states[typ2]
                i_w_stop                = i_w + n_features * n_states1 * n_states2
                i_states2_stop          = i_states2 + n_states2
                
                self._cache_pairwise_potentials.append( (n_features
                                                         , n_states1, n_states2, i_states1, i_states1_stop, i_states2, i_states2_stop
                                                         , i_w, i_w_stop) )
                
                i_w, i_states2 = i_w_stop, i_states2_stop 
            i_states1 = i_states1_stop
        
    def _get_pairwise_potentials(self, x, w):
        """Computes pairwise potentials for x and w.

        Parameters
        ----------
        x : tuple
            Instance Representation.

        w : ndarray, shape=(size_joint_feature,)
            Weight vector for CRF instance.

        Returns
        -------
        pairwise : ndarray, shape=(n_states, n_states)
            Pairwise weights.
        """
        self._check_size_w(w)
        #self._check_size_x(x) #call initialize once and only once before!!
        
        l_edge_features  = self._get_edge_features(x)
        l_edge_nb = [0 if ef is None else ef.shape[0] for ef in l_edge_features]
        n_edges_total    = sum(l_edge_nb)
        
        wpw = w[self.size_unaries:]
        a_edges_states_states = np.zeros((n_edges_total, self._n_states, self._n_states), dtype=w.dtype)

#         i_w, i_edges, i_states1, i_states2 = 0, 0, 0, 0
# #         for (typ1, typ2), edge_features, edgetype_start_index in zip(self._iter_type_pairs(), l_edge_features, self._l_edgetype_start_index):
#         for (typ1, typ2), edge_features in zip(self._iter_type_pairs(), l_edge_features):
#             if edge_features is None: continue
#              
#             n_edges, n_features     = edge_features.shape
#             n_states1               = self.l_n_states[typ1]
#             n_states2               = self.l_n_states[typ2]
#             i_w_stop                = i_w + self.a_n_edge_features[typ1,typ2] * n_states1 * n_states2
#             i_edges_stop            = i_edges + n_edges
#             i_states1_stop          = i_states1 + n_states1
#             i_states2_stop          = i_states2 + n_states2
#              
#             pw_typ_typ = wpw[i_w:i_w_stop].reshape(n_features, -1) # n_states1*n_states2 x nb_feat
#             pot_typ_typ = np.dot(edge_features, pw_typ_typ).reshape(n_edges, n_states1, n_states2)
#             print "pot_typ_typ.shape ", pot_typ_typ.shape
#             a_edges_states_states[ i_edges:i_edges_stop, i_states1:i_states1_stop , i_states2:i_states2_stop ] = pot_typ_typ
#              
#             i_w, i_edges, i_states1, i_states2 = i_w_stop, i_edges_stop, i_states1_stop, i_states2_stop 

        i_edges = 0
        #print map(len, [self._cache_pairwise_potentials, l_edge_features, l_edge_nb])
        for ((n_features, n_states1, n_states2, i_states1, i_states1_stop, i_states2, i_states2_stop, i_w, i_w_stop)
             , edge_features, n_edges) in zip(self._cache_pairwise_potentials, l_edge_features, l_edge_nb):
             
            i_edges_stop            = i_edges + n_edges
             
            if not edge_features is None: 
                pw_typ_typ = wpw[i_w:i_w_stop].reshape(n_features, -1) # n_states1*n_states2 x nb_feat
                pot_typ_typ = np.dot(edge_features, pw_typ_typ).reshape(n_edges, n_states1, n_states2)
#                 print i_states1,i_states1_stop , i_states2,i_states2_stop, n_states1, n_states2
#                 print "a_edges_states_states.shape ", a_edges_states_states.shape
#                 print "a_edges_states_states[  ].shape ", a_edges_states_states[ i_edges:i_edges_stop, i_states1:i_states1_stop , i_states2:i_states2_stop ].shape
                a_edges_states_states[ i_edges:i_edges_stop, i_states1:i_states1_stop , i_states2:i_states2_stop ] = pot_typ_typ
 
            i_edges = i_edges_stop 
                    
        return a_edges_states_states.reshape(n_edges_total, self._n_states, self._n_states)

    def _block_ravel(self, a, lij):
        """
        Ravel the array block by block
        """
        li, lj  = zip(*lij)
        return np.hstack( [a[i0:i1,j0:j1].ravel()
                                   for (i0, i1), (j0,j1)
                                   in zip( zip(li, li[1:]), zip(lj, lj[1:]) ) 
                                   ])
 
    def joint_feature(self, x, y):
        """Feature vector associated with instance (x, y).

        Feature representation joint_feature, such that the energy of the configuration
        (x, y) and a weight vector w is given by np.dot(w, joint_feature(x, y)).

        Parameters
        ----------
        x : tuple
            Input representation.

        y : list of ndarrays or some tuple (internal use!)
            Either y is a list of a integral ndarrays, giving a complete labeling for x.
            Or it is the result of a linear programming relaxation. In this
            case, ``y=(unary_marginals, pariwise_marginals)``.

        Returns
        -------
        p : ndarray, shape (size_joint_feature,)
            Feature vector associated with state (x, y).

        """
#         print "x=", `x`
#         print "y=", `y`
        self._check_size_x(x)   #call initialize once!
        l_node_features = self._get_node_features(x)
        l_edges, l_edge_features = self._get_edges(x), self._get_edge_features(x)
        l_n_nodes = [len(o) for o in self._get_node_features(x, True)]
        l_n_edges = [edges.shape[0] for edges in self._get_edges(x, True)]
        n_nodes = sum(l_n_nodes)
        n_edges = sum(l_n_edges)
        if False:
            print
            print type(y)
            print "l_n_nodes = ", l_n_nodes
            for nf in l_node_features: print "nf.shape ", None if nf is None else nf.shape, 
            print
            print "l_n_edges = ", l_n_edges
            for ef in l_edge_features: print "ef.shape ", None if ef is None else ef.shape, 
            print 
        if isinstance(y, tuple):
            #print "y=", `y`
            # y is result of relaxation, tuple of unary and pairwise marginals
            unary_marginals, pw = y
            unary_marginals = unary_marginals.reshape(n_nodes, self._n_states)
        else:
            self._check_size_xy(x, y)
            #make one hot encoding
            #each type is assigned a range of columns, each starting at self._a_state_startindex_by_typ[ <typ> ]
            #in the arnge column I is for state i of that type
            unary_marginals = np.zeros((n_nodes, self._n_states), dtype=np.int)
            i_start = 0
            #print self.l_n_states, self._l_type_startindex, y
#             print "l_node_features shapes", map(lambda x: x.shape, l_node_features)
#             print "y.shape", y.shape
#             print "y", y.ravel()
            for node_features, typ_start_index in zip(l_node_features, self._l_type_startindex):
                if node_features is None: continue
                i_stop = i_start + node_features.shape[0]
#                 print "typ_start_index ", typ_start_index
#                 print "y. ", y.shape, i_start, i_stop
#                 print  y[i_start:i_stop].ravel()
                unary_marginals[ :, typ_start_index + y[i_start:i_stop] ] 
                unary_marginals[ np.ogrid[i_start:i_stop] 
                                , typ_start_index + y[i_start:i_stop] 
                                ] = 1
                i_start = i_stop
            #print "--- unary_marginals \n", `unary_marginals`
            
            ## pairwise
            #same thing, but the type of an edge is a pair of node types 
            pw = np.zeros((n_edges, self._n_states ** 2))
            node_offset_by_typ = np.cumsum([0]+[0 if n is None else n.shape[0] for n in x[0]])
            i_start = 0
            for (typ1, typ2), edges, edgetype_start_index in zip(self._iter_type_pairs(), l_edges, self._l_edgetype_start_index):
                if edges is None: continue
                #the label of those pairs of nodes
                y1 = y[node_offset_by_typ[typ1] + edges[:,0]]
                y2 = y[node_offset_by_typ[typ2] + edges[:,1]]
                #set the 1s where they should
                i_stop = i_start + edges.shape[0]
                pw[ np.ogrid[i_start:i_stop]
                   , edgetype_start_index + self.l_n_states[typ2] * y1 + y2
                   ] = 1
                i_start = i_stop
            #print "--- pw = \n", `pw`
            assert i_start == n_edges
            
        #UNARY
        #assign the feature of each node t the right range of column according to the node type
        all_node_features = np.zeros((n_nodes, self._n_features))
        i_start = 0
        for (_a_feature_slice, node_features) in zip(self._a_feature_slice_by_typ, l_node_features):
            i_stop = i_start + node_features.shape[0]
            all_node_features[ i_start:i_stop
                              , _a_feature_slice] = node_features
            i_start = i_stop
        assert i_start == n_nodes
        #print "--- all_node_features =\n", `all_node_features`
        unaries_acc = np.dot(unary_marginals.T, all_node_features)   # node_states x sum_of_features matrix
        #print "--- unaries_acc =\n", `unaries_acc`
        
        #assign the edges feature to the right range of columns, depending on edge type
        all_edge_features = np.zeros( (n_edges, self._n_edge_features) )
        i_start = 0
        i_col_start = 0
        for edge_features in l_edge_features:
            if edge_features is None: continue
            nb_edges, nb_features = edge_features.shape
            i_stop     = i_start     + nb_edges
            i_col_stop = i_col_start + nb_features
            all_edge_features[ i_start:i_stop
                              , i_col_start:i_col_stop ] = edge_features
            i_col_start = i_col_stop
            i_start     = i_stop
        #print "--- all_edge_features =\n", `all_edge_features`

        #print all_edge_features.T.shape, pw.shape
            
        pairwise_acc = np.dot(all_edge_features.T, pw)      # sum_of_features x edge_states
        
        
#         print '-'*30
#         print np.dot(pw.T, all_edge_features).T
#         print '-'*30
        
        #easier to read... :-( pairwise_acc = np.dot(pw.T, all_edge_features)      # sum_of_features x edge_states
        #print "--- pairwise_acc.shape = ", pairwise_acc.shape        
        #print "--- pairwise_acc =\n", `pairwise_acc`        

#         for i in self.symmetric_edge_features:
#             pw_ = pw[i].reshape(self.n_states, self.n_states)
#             pw[i] = (pw_ + pw_.T).ravel() / 2.
# 
#         for i in self.antisymmetric_edge_features:
#             pw_ = pw[i].reshape(self.n_states, self.n_states)
#             pw[i] = (pw_ - pw_.T).ravel() / 2.


#         print `unaries_acc`
#         print "unaries_acc.size = ", unaries_acc.size

        #we need to linearize it, while keeping only meaningful data
        unaries_acc_ravelled = self._block_ravel(unaries_acc, [(0,0)]+zip(np.cumsum(self.l_n_states), np.cumsum(self.l_n_features)))
        #print "--- unaries_acc_ravelled =\n", `unaries_acc_ravelled`
        assert len(unaries_acc_ravelled) == self.size_unaries

        L1 = np.cumsum(self.a_n_edge_features.ravel())
        L2 = np.cumsum([self.l_n_states[typ1] * self.l_n_states[typ2] for typ1, typ2 in self._iter_type_pairs() ])
#       easier to read...      aux=L1; L1=L2; L2=aux
        pairwise_acc_ravelled = self._block_ravel(pairwise_acc, [(0,0)]+zip(L1,L2))

        #print "--- pairwise_acc_ravelled =\n", `pairwise_acc_ravelled`
        assert len(pairwise_acc_ravelled) == self.size_pairwise
        
#         print `unaries_acc_ravelled`
#         print "unaries_acc_ravelled.size = ", unaries_acc_ravelled.size
#         print "unaries_acc_ravelled.shape = ", unaries_acc_ravelled.shape
        
#         print "pairwise_acc_ravelled.size = ", pairwise_acc_ravelled.size
#         print "pairwise_acc_ravelled.shape = ", pairwise_acc_ravelled.shape
#         print `pairwise_acc_ravelled`
        joint_feature_vector = np.hstack([unaries_acc_ravelled, pairwise_acc_ravelled])
        
        assert joint_feature_vector.shape[0] == self.size_joint_feature, (joint_feature_vector.shape[0], self.size_joint_feature)
        return joint_feature_vector


