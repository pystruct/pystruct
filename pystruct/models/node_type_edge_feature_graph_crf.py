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
import random

from ..inference import inference_dispatch
from .utils import loss_augment_unaries

from .typed_crf import TypedCRF, InconsistentLabel


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
    
    NOTE: there should always be at least 1 feature for any pairs of types with some edge of that type in the graph.
    Said differently, if you put 0 somewhere in that matrix, do not create any egde corresponding to that type of edge!!  
    
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
        
        NOTE: for now, a typ1, typ2 type of edge with 0 features is simply ignored. While it could get a state x state matrix of weights
        """
        if self.l_n_features:
            self.size_unaries = sum(  n_states * n_features for n_states, n_features in zip(self.l_n_states, self.l_n_features) )
        
            self.size_pairwise = 0  #detailed non-optimized computation to make things clear
            for typ1,typ2 in self._iter_type_pairs():
                self.size_pairwise += self.a_n_edge_features[typ1,typ2] * self.l_n_states[typ1] * self.l_n_states[typ2]

            self.size_joint_feature = self.size_unaries + self.size_pairwise
        
    def __repr__(self):
        return ("%s(n_states: %s, inference_method: %s, n_features: %s, "
                "n_edge_features: %s)"
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
        return True

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
        pairwise : ndarray, shape=(n_edges, n_states, n_states)
            Pairwise weights.
        """
        self._check_size_w(w)
        #self._check_size_x(x) #call initialize once and only once before!!
        
        l_edge_features  = self._get_edge_features(x)
        l_edge_nb = [0 if ef is None else ef.shape[0] for ef in l_edge_features]
        n_edges_total    = sum(l_edge_nb)
        
        wpw = w[self.size_unaries:]
        a_edges_states_states = np.zeros((n_edges_total, self._n_states, self._n_states), dtype=w.dtype)

        i_edges = 0
        for ((n_features, n_states1, n_states2, i_states1, i_states1_stop, i_states2, i_states2_stop, i_w, i_w_stop)
             , edge_features, n_edges) in zip(self._cache_pairwise_potentials, l_edge_features, l_edge_nb):
             
            i_edges_stop            = i_edges + n_edges
             
            if not edge_features is None: 
                pw_typ_typ = wpw[i_w:i_w_stop].reshape(n_features, -1) # n_states1*n_states2 x nb_feat
                pot_typ_typ = np.dot(edge_features, pw_typ_typ).reshape(n_edges, n_states1, n_states2)
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
            for node_features, typ_start_index in zip(l_node_features, self._l_type_startindex):
                if node_features is None: continue
                i_stop = i_start + node_features.shape[0]
                unary_marginals[ np.ogrid[i_start:i_stop] 
                                , y[i_start:i_stop] 
                                ] = 1
                i_start = i_stop
            
            ## pairwise
            #same thing, but the type of an edge is a pair of node types 
            pw = np.zeros((n_edges, self._n_states ** 2))
            node_offset_by_typ = np.cumsum([0]+[0 if n is None else n.shape[0] for n in x[0]])
            i_start = 0
            for (typ1, typ2), edges, edgetype_start_index in zip(self._iter_type_pairs(), l_edges, self._l_edgetype_start_index):
                if edges is None: continue
                y1 = y[node_offset_by_typ[typ1] + edges[:,0]] - self._l_type_startindex[typ1]
                assert (0<=y1).all() and (y1 <= self.l_n_states[typ1]).all()
                y2 = y[node_offset_by_typ[typ2] + edges[:,1]] - self._l_type_startindex[typ2]
                assert (0<=y2).all() and (y2 <= self.l_n_states[typ2]).all()
                #set the 1s where they should
                i_stop = i_start + edges.shape[0]
                pw[ np.ogrid[i_start:i_stop]
                   , edgetype_start_index + self.l_n_states[typ2] * y1 + y2
                   ] = 1
                i_start = i_stop
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
        unaries_acc = np.dot(unary_marginals.T, all_node_features)   # node_states x sum_of_features matrix
        
        #assign the edges feature to the right range of columns, depending on edge type
        all_edge_features = np.zeros( (n_edges, self._n_edge_features) )
        i_start = 0
        i_col_start = 0
        for edge_features, n_feat in zip(l_edge_features, self.a_n_edge_features.ravel()):
            i_col_stop = i_col_start + n_feat
            
            if not edge_features is None: 
                nb_edges = edge_features.shape[0]
                i_stop     = i_start     + nb_edges
                all_edge_features[ i_start:i_stop
                                  , i_col_start:i_col_stop ] = edge_features
                i_start     = i_stop
            i_col_start = i_col_stop
        
#         if False:
#             np.set_printoptions(precision=3, linewidth=9999)
#             print "all_edge_features (edgexfeat) \n", `all_edge_features`
#             print "pw (edgexstate)\n", `pw`
        pairwise_acc = np.dot(all_edge_features.T, pw)      # sum_of_features x edge_states

# This forced symetry / antisymetry is not supported for now
#         for i in self.symmetric_edge_features:
#             pw_ = pw[i].reshape(self.n_states, self.n_states)
#             pw[i] = (pw_ + pw_.T).ravel() / 2.
# 
#         for i in self.antisymmetric_edge_features:
#             pw_ = pw[i].reshape(self.n_states, self.n_states)
#             pw[i] = (pw_ - pw_.T).ravel() / 2.

        #we need to linearize it, while keeping only meaningful data
        unaries_acc_ravelled = self._block_ravel(unaries_acc, [(0,0)]+zip(np.cumsum(self.l_n_states), np.cumsum(self.l_n_features)))
        assert len(unaries_acc_ravelled) == self.size_unaries

        L1 = np.cumsum(self.a_n_edge_features.ravel())
        L2 = np.cumsum([self.l_n_states[typ1] * self.l_n_states[typ2] for typ1, typ2 in self._iter_type_pairs() ])
        pairwise_acc_ravelled = self._block_ravel(pairwise_acc, [(0,0)]+zip(L1,L2))

        assert len(pairwise_acc_ravelled) == self.size_pairwise
        joint_feature_vector = np.hstack([unaries_acc_ravelled, pairwise_acc_ravelled])
        
        assert joint_feature_vector.shape[0] == self.size_joint_feature, (joint_feature_vector.shape[0], self.size_joint_feature)

        return joint_feature_vector


    def loss_augmented_inference(self, x, y, w, relaxed=False,
                                 return_energy=False):
        """Loss-augmented Inference for x relative to y using parameters w.

        Finds (approximately)
        armin_y_hat np.dot(w, joint_feature(x, y_hat)) + loss(y, y_hat)
        using self.inference_method.


        Parameters
        ----------
        x : tuple
            Instance of a graph with unary evidence.
            x=(unaries, edges)
            unaries are an nd-array of shape (n_nodes, n_features),
            edges are an nd-array of shape (n_edges, 2)

        y : ndarray, shape (n_nodes,)
            Ground truth labeling relative to which the loss
            will be measured.

        w : ndarray, shape=(size_joint_feature,)
            Parameters for the CRF energy function.

        relaxed : bool, default=False
            Whether relaxed inference should be performed.
            Only meaningful if inference method is 'lp' or 'ad3'.
            By default fractional solutions are rounded. If relaxed=True,
            fractional solutions are returned directly.

        return_energy : bool, default=False
            Whether to return the energy of the solution (x, y) that was found.

        Returns
        -------
        y_pred : ndarray or tuple
            By default an inter ndarray of shape=(n_nodes)
            of variable assignments for x is returned.
            If ``relaxed=True`` and inference_method is ``lp`` or ``ad3``,
            a tuple (unary_marginals, pairwise_marginals)
            containing the relaxed inference result is returned.
            unary marginals is an array of shape (n_nodes, n_states),
            pairwise_marginals is an array of
            shape (n_states, n_states) of accumulated pairwise marginals.

        """
#         print "y.shape ", y.shape
        self.inference_calls += 1
        self._check_size_w(w)
        unary_potentials    = self._get_unary_potentials(x, w)
        pairwise_potentials = self._get_pairwise_potentials(x, w)
        flat_edges = self._index_all_edges(x)
        
        loss_augment_unaries(unary_potentials, np.asarray(y), self.class_weight)

        if self.inference_method == "ad3+":
            l_n_nodes = [nf.shape[0] for nf in self._get_node_features(x, True)]
            nodetype_data = (l_n_nodes, self.l_n_states) #the type of the nodes, the number of state by type
    
            Y_pred = inference_dispatch(unary_potentials, pairwise_potentials, flat_edges,
                                      self.inference_method, relaxed=relaxed,
                                      return_energy=return_energy,
                                      nodetype=nodetype_data)
            #with ad3+ this should never occur
            if not isinstance(Y_pred, tuple): assert self._check_size_xy(x, Y_pred), "Internal error in AD3+: inconsistent labels"
        else:
            Y_pred = inference_dispatch(unary_potentials, pairwise_potentials, flat_edges,
                                      self.inference_method, relaxed=relaxed,
                                      return_energy=return_energy)
                                        #no nodetype parameter!
            #we may have inconsistent labels!
            try:
                if not isinstance(Y_pred, tuple): self._check_size_xy(x, Y_pred)
            except InconsistentLabel:
                #the inference engine predicted inconsistent labels
                Y_pred = self.fix_Y_at_random(x, Y_pred)
            
        #if self.inference_calls % 1000 == 0: print "%d inference calls"%self.inference_calls
                    
        return Y_pred
    
    def fix_Y_at_random(self, x, Y_pred):
        print "\tY is BAD, FIXING IT AT RANDOM", `Y_pred`
        
        l_node_features = self._get_node_features(x, True)
        i_start = 0              
        for typ, (nf, n_states) in enumerate(zip(l_node_features, self.l_n_states)):
            nb_nodes = nf.shape[0]
            if nb_nodes:
                Y_typ = Y_pred[i_start:i_start+nb_nodes]
                typ_start = self._l_type_startindex[typ]
                typ_end   = self._l_type_startindex[typ+1]
                if np.min(Y_typ) < typ_start  or typ_end <= np.max(Y_typ):
                    for i in range(nb_nodes):
                        if Y_pred[i_start+i] < typ_start  or typ_end <= Y_pred[i_start+i]:  Y_pred[i_start+i] = random.randint(typ_start, typ_end-1)
            i_start = i_start + nb_nodes      
        self._check_size_xy(x, Y_pred)  
        return Y_pred
        
    def inference(self, x, w, relaxed=False, return_energy=False, constraints=None):
        """Inference for x using parameters w.

        Finds (approximately)
        armin_y np.dot(w, joint_feature(x, y))
        using self.inference_method.


        Parameters
        ----------
        x : tuple
            Instance of a graph with unary evidence.
            x=(unaries, edges)
            unaries are an nd-array of shape (n_nodes, n_states),
            edges are an nd-array of shape (n_edges, 2)

        w : ndarray, shape=(size_joint_feature,)
            Parameters for the CRF energy function.

        relaxed : bool, default=False
            Whether relaxed inference should be performed.
            Only meaningful if inference method is 'lp' or 'ad3'.
            By default fractional solutions are rounded. If relaxed=True,
            fractional solutions are returned directly.

        return_energy : bool, default=False
            Whether to return the energy of the solution (x, y) that was found.

        constraints : None or list, default=False
            hard logic constraints, if any
            
        Returns
        -------
        y_pred : ndarray or tuple
            By default an inter ndarray of shape=(width, height)
            of variable assignments for x is returned.
            If ``relaxed=True`` and inference_method is ``lp`` or ``ad3``,
            a tuple (unary_marginals, pairwise_marginals)
            containing the relaxed inference result is returned.
            unary marginals is an array of shape (width, height, n_states),
            pairwise_marginals is an array of
            shape (n_states, n_states) of accumulated pairwise marginals.

        """
        self._check_size_w(w)
        self.inference_calls += 1
        self.initialize(x)
        unary_potentials    = self._get_unary_potentials(x, w)
        pairwise_potentials = self._get_pairwise_potentials(x, w)
        flat_edges = self._index_all_edges(x)

        l_n_nodes = [nf.shape[0] for nf in self._get_node_features(x, True)]
        nodetype_data=(l_n_nodes, self.l_n_states) #the type of the nodes, the number of state by type

        if self.inference_method == "ad3+":
            #preferred method for TypedCRF inferences (called by the 'predict' method of the learner)
            Y_pred = inference_dispatch(unary_potentials, pairwise_potentials, flat_edges,
                                          self.inference_method, relaxed=relaxed,
                                          return_energy=return_energy,
                                          constraints=constraints,
                                          nodetype=nodetype_data,
                                          inference_exception=self.inference_exception)             #<--
        else:
            if constraints:
                Y_pred = inference_dispatch(unary_potentials, pairwise_potentials, flat_edges,
                                          self.inference_method, relaxed=relaxed,
                                          return_energy=return_energy,
                                          constraints=constraints)              #<--
            else:
                Y_pred = inference_dispatch(unary_potentials, pairwise_potentials, flat_edges,
                                          self.inference_method, relaxed=relaxed,
                                          return_energy=return_energy)             #<--

        try:
            if not isinstance(Y_pred, tuple): self._check_size_xy(x, Y_pred)
        except:
            Y_pred = self.fix_Y_at_random(x, Y_pred)
            
        if not isinstance(Y_pred, tuple): self._check_size_xy(x, Y_pred)
        
        return Y_pred
