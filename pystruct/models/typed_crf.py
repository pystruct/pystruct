import numpy as np

from .base import StructuredModel
from ..inference import inference_dispatch, get_installed
from .utils import loss_augment_unaries
from numpy import dtype


class TypedCRF(StructuredModel):
    """Abstract base class"""
    def __init__(self
                 , n_types                  #how many node type?
                 , l_n_states               #how many labels   per node type?
                 , l_n_features             #how many features per node type?
                 , l_class_weight=None):    #class_weight      per node type or None           <list of array-like> or None
        if len(l_n_states)   != n_types:
            raise ValueError("Expected 1 number of states per node type.")
        if l_n_features != None and len(l_n_features) != n_types:
            raise ValueError("Expected 1 number pf features per node type.")
        self.n_types      = n_types
        self.l_n_states   = l_n_states
        self._n_states    = sum(l_n_states)     #total number of states
        self.l_n_features = l_n_features
        self._n_features  = sum(self.l_n_features)   #total number of (node) features

        # check that ad3 is installed
        inference_method = get_installed(['ad3'])
        if not inference_method: raise Exception("ERROR: this model class requires AD3.")
        self.inference_method = inference_method[0]
        self.inference_calls = 0
        
        #class weights:
        # either we get class weights for all types of nodes, or for none of them!
        if l_class_weight:
            if len(l_class_weight) != self.n_types:
                raise ValueError("Expected 1 class weight list per node type.")
            for i, n_states in enumerate(self.l_n_states):
                if len(l_class_weight[i]) != n_states:
                    raise ValueError("Expected 1 class weight per state per node type. Wrong for l_class_weight[%d]"%i)
                    
            #class weights are computed by type and simply concatenated
            self.class_weight = np.hstack([np.array(class_weight) for class_weight in l_class_weight])
        else:
            n_things = sum(self.l_n_states)
            self.class_weight = np.ones(n_things)

        self._set_size_joint_feature()

        #internal stuff
        #when putting features in a single sequence, index of 1st state for type i
        self._l_type_startindex = [ sum(self.l_n_states[:i]) for i in range(self.n_types)]

        #when putting states in a single sequence, index of 1st feature for type i (is at Ith position)
        #we store the slice objects
        self._a_feature_slice_by_typ = np.array([ slice(sum(self.l_n_features[:i]), sum(self.l_n_features[:i+1])) for i in range(self.n_types)])

        #when putting edge states in a single sequence, index of 1st feature of an edge of type (typ1, typ2)
        self._l_edgetype_start_index = [] 
        i_start = 0
        for typ1_n_states in self.l_n_states:
            for typ2_n_states in self.l_n_states:
                self._l_edgetype_start_index.append(i_start)
                i_start += typ1_n_states*typ2_n_states 
        self._l_edgetype_start_index.append(i_start)
        assert i_start == self._n_states**2
        

    def _set_size_joint_feature(self):
        """
        We have:
        - 1 weight per node feature per label per node type
        """
        self.size_unaries = sum(  n_states * n_features for n_states, n_features in zip(self.l_n_states, self.l_n_features) )
        self.size_joint_feature = self.size_unaries

    def __repr__(self):
        return ("%s(n_states: %s, inference_method: %s)"
                % (type(self).__name__, self.l_n_states,
                   self.inference_method))

    def _check_size_x(self, x):
        l_nodes = self._get_node_features(x)
        
        #node_features are [  i_in_typ -> features ]
        l_features = self._get_node_features(x)
        if len(l_features) != self.n_types:
            raise ValueError("Expected one node feature array per node type.")
        
        for typ, typ_features in enumerate(l_features):
            if typ_features.shape[1] != self.l_n_features[typ]:
                raise ValueError("Expected %d features for type %d"%(self.l_n_features[typ], typ))

        #edges
        l_edges = self._get_edges(x)
        for edges in l_edges:
            if edges is None: continue
            if edges.ndim != 2:
                raise ValueError("Expected a 2 dimensions edge arrays")
            if edges.shape[1] != 2:
                raise ValueError("Expected 2 columns in edge arrays")

        for typ1,typ2 in self._iter_type_pairs():
            edges = self._get_edges_by_type(x, typ1, typ2) 
        
            if edges is None or len(edges) == 0: continue
            #edges should point to valid node indices
            nodes1, nodes2 = edges[:,0], edges[:,1]
            if min(nodes1) < 0 or min(nodes2) < 0:
                raise ValueError("At least one edge points to negative and therefore invalid node index")
            if max(nodes1) >= l_nodes[typ1].shape[0] or max(nodes2) > l_nodes[typ2].shape[0]:
                raise ValueError("At least one edge points to non-existing node index")
    
    def _check_size_y(self, x, y):
        
        if not isinstance(y, list): 
            raise ValueError("Y must be a list of arrays")
        
        l_features = self._get_node_features(x)
        
        for typ, (features, y_typ) in enumerate(zip(l_features, y)):
            if not isinstance(y_typ, np.ndarray): 
                raise ValueError("Y must be a list of arrays")
            if features.shape[0] != len(y_typ):
                raise ValueError("Node of type %d: Expected %d labels not %d"%(typ, features.shape[0], len(y_typ)))
            
            if min(y_typ) < 0 or max(y_typ) >=self.l_n_states[typ]:
                    raise ValueError("Type %d: Some invalid label")

    def _get_node_features(self, x, bClean=False):
        if bClean:
            return [ np.empty((0,0)) if node_features is None or len(node_features)==0 else node_features for node_features in x[0]]
        else:
            return x[0]
    def _get_node_features_by_type(self, x, typ):
        return x[0][typ]
    def _get_edges(self, x, bClean=False):
        if bClean:
            return [ np.empty((0,0)) if edges is None or len(edges)==0 else edges for edges in x[1]]
        else:
            return x[1]
    def _get_edges_by_type(self, x, typ1, typ2):
        return x[1][typ1*self.n_types+typ2] 

    def _iter_type_pairs(self):
        for typ1 in range(self.n_types):
            for typ2 in range(self.n_types):
                yield (typ1, typ2)
        raise StopIteration
    
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
        self.inference_calls += 1
        self._check_size_w(w)
        unary_potentials = self._get_unary_potentials(x, w)
        pairwise_potentials = self._get_pairwise_potentials(x, w)
        edges = self._get_edges(x)
        loss_augment_unaries(unary_potentials, np.asarray(y), self.class_weight)

        return inference_dispatch(unary_potentials, pairwise_potentials, edges,
                                  self.inference_method, relaxed=relaxed,
                                  return_energy=return_energy)

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
        unary_potentials = self._get_unary_potentials(x, w)
        pairwise_potentials = self._get_pairwise_potentials(x, w)
        edges = self._get_edges(x)

        if constraints:
            return inference_dispatch(unary_potentials, pairwise_potentials, edges,
                                      self.inference_method, relaxed=relaxed,
                                      return_energy=return_energy, constraints=constraints)
        else:
            return inference_dispatch(unary_potentials, pairwise_potentials, edges,
                                      self.inference_method, relaxed=relaxed,
                                      return_energy=return_energy)