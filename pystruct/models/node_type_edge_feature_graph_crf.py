# -*- coding: utf-8 -*-

"""
    Pairwise CRF with features/strength associated to each edge and different
    types of nodes

    JL. Meunier

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

    Developed  for the EU project READ. The READ project has received funding
    from the European Union's Horizon 2020 research and innovation programme
    under grant agreement No 674943.

"""
import numpy as np
import random

from ..inference import inference_dispatch
from .utils import loss_augment_unaries

from .typed_crf import TypedCRF, InconsistentLabel


class NodeTypeEdgeFeatureGraphCRF(TypedCRF):
    """
    Pairwise CRF with features/strength associated to each edge and different
    types of nodes

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

    a_n_edge_features: an array of shape (n_types, n_types) giving the number
        of features per pair of types

    NOTE: there should always be at least 1 feature for any pairs of types
        which has some edge in the graph.
        To mimic GraphCRF, pass 1 and make a constant feature of 1.0 for all
        those edges.

    class_weight : None, or list of array-like (ndim=1)
        Class weights. If a list of array-like is passed, the Ith one must have
        length equal to l_n_states[i]
        None means equal class weights (across node types)

    X and Y
    -------
    Node features are given as a list of n_types arrays of shape
            (n_type_nodes, n_type_features):
        - n_type_nodes is the number of nodes of that type
        - n_type_features is the number of features for this type of node

    Edges are given as a list of n_types x n_types arrays of shape
            (n_type_edges, 2).
        Columns are resp.: node index (in corresponding node type), node index
        (in corresponding node type)

    Edge features are given as a list of n_types x n_types arrays of shape
            (n_type_type_edge, n_type_type_edge_features)
        - n_type_type_edge is the number of edges of type type_type
        - n_type_type_edge_features is the number of features for edge of type
            type_type

    An instance ``X`` is represented as a tuple ``([node_features, ..]
            , [edges, ..], [edge_features, ..])``

    Labels ``Y`` are given as one array of shape (n_nodes)
        Labels are numbered from 0 so that each label across types is encoded
        by a unique integer.

        Look at flattenY and unflattentY if you want to pass/obtain list of
        labels per type, with first label of each type being encoded by 0

    """

    def __init__(self,
                 n_types,                  # how many node type?
                 l_n_states,               # how many labels   per node type?
                 l_n_features,             # how many features per node type?
                 a_n_edge_features,        # how many features per edge type?
                 inference_method="ad3",
                 l_class_weight=None):    # class_weight per node type or None
                                            # <list of array-like> or None

        # how many features per node type X node type?
        # <array-like> (MUST be symmetric!)
        self.a_n_edge_features = np.array(a_n_edge_features)
        if self.a_n_edge_features.shape != (n_types, n_types):
            raise ValueError("Expected a feature number matrix for edges of "
                             "shape (%d, %d), got "
                             "%s." % (n_types, n_types,
                                      self.a_n_edge_features.shape))
        self.a_n_edge_features = self.a_n_edge_features.reshape(n_types,
                                                                n_types)
        if not (self.a_n_edge_features == self.a_n_edge_features.T).all():
            raise ValueError("Expected a symmetric array of edge feature "
                             "numbers")

        # number of (edge) features per edge type
        self.l_n_edge_features = self.a_n_edge_features.ravel()
        # total number of (edge) features
        self._n_edge_features = self.a_n_edge_features.sum(axis=None)

        TypedCRF.__init__(self, n_types, l_n_states, l_n_features,
                          inference_method=inference_method,
                          l_class_weight=l_class_weight)

        self._get_pairwise_potentials_initialize()

    def _set_size_joint_feature(self):
        """
        We have:
        - 1 weight per node feature per label per node type
        - 1 weight per edge feature per label of node1 type, per label of node2
            type

        NOTE: for now, a typ1, typ2 type of edge with 0 features is simply
                ignored. While it could get a state x state matrix of weights
        """
        if self.l_n_features:
            self.size_unaries = sum(n_states * n_features for n_states,
                                    n_features in zip(self.l_n_states,
                                                      self.l_n_features))

            # detailed non-optimized computation to make things clear
            self.size_pairwise = 0
            for typ1, typ2 in self._iter_type_pairs():
                self.size_pairwise += self.a_n_edge_features[typ1, typ2]\
                                    * self.l_n_states[typ1]\
                                    * self.l_n_states[typ2]

            self.size_joint_feature = self.size_unaries + self.size_pairwise

    def __repr__(self):
        return ("%s(n_states: %s, inference_method: %s, n_features: %s, "
                "n_edge_features: %s)"
                % (type(self).__name__, self.l_n_states, self.inference_method,
                   self.l_n_features, self.a_n_edge_features))

    def _check_size_x(self, x):
        l_edges = self._get_edges(x)
        if len(l_edges) != self.n_types**2:
            raise ValueError("Expected %d edge arrays "
                             "or None" % (self.n_types**2))
        l_edge_features = self._get_edge_features(x)
        if len(l_edge_features) != self.n_types**2:
            raise ValueError("Expected %d edge feature arrays "
                             "or None" % (self.n_types**2))

        TypedCRF._check_size_x(self, x)

        # check that we have in total 1 feature vector per edge
        for edges, edge_features in zip(l_edges, l_edge_features):
            if edges is None or edge_features is None:
                if edges is None and edge_features is None:
                    continue
                if edges is None:
                    raise ValueError("Empty edge array but non empty "
                                     "edge-feature array, for same type of "
                                     "edge")
                else:
                    raise ValueError("Empty edge-feature array but non empty "
                                     "edge array, for same type of edge")
            if edge_features.ndim != 2:
                raise ValueError("Expected a 2 dimensions edge feature arrays")
            if len(edges) != len(edge_features):
                raise ValueError("Edge and edge feature matrices must have "
                                 "same size in 1st dimension")

        # check edge feature size
        for typ1, typ2 in self._iter_type_pairs():
            edge_features = l_edge_features[typ1*self.n_types+typ2]
            if edge_features is None:
                continue
            if edge_features.shape[1] != self.a_n_edge_features[typ1, typ2]:
                raise ValueError("Types %d x %d: bad number of edge features. "
                                 "expected %d "
                                 "got %d" % (typ1, typ2,
                                             self.a_n_edge_features[typ1,
                                                                    typ2],
                                             edge_features.shape[1]))
        return True

    def _get_edge_features(self, x):
        # we replace None by empty array with proper shape
        return [np.empty((0, _n_feat))
                if _ef is None
                else _ef
                for _ef, _n_feat in zip(x[2], self.l_n_edge_features)]

    def _get_pairwise_potentials_initialize(self):
        """
        Putting in cache the params required to build the pairwise potentials
        given x and w
        """
        self._cache_pairwise_potentials = list()

        i_w, n_states1, i_states1 = 0, 0, 0

        for typ1 in range(self.n_types):
            n_states1 = self.l_n_states[typ1]
            i_states1_stop = i_states1 + n_states1
            n_states2, i_states2 = 0, 0
            for typ2 in range(self.n_types):
                n_features = self.a_n_edge_features[typ1, typ2]
                n_states2 = self.l_n_states[typ2]
                i_w_stop = i_w + n_features * n_states1 * n_states2
                i_states2_stop = i_states2 + n_states2

                self._cache_pairwise_potentials.append((n_features,
                                                        n_states1, n_states2,
                                                        i_states1,
                                                        i_states1_stop,
                                                        i_states2,
                                                        i_states2_stop,
                                                        i_w, i_w_stop))

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
        pairwise: list of pairwise weights of shape:
            (n_edges, n_states_typA, n_states_typB)

        """
        self._check_size_w(w)

        l_edge_features = self._get_edge_features(x)
        wpw = w[self.size_unaries:]

        l_pairwise_potentials = []

        i_w = 0
        for (typ1, typ2), edge_features in zip(self._iter_type_pairs(),
                                               l_edge_features):
            n_edges, n_features = edge_features.shape
            n_states1 = self.l_n_states[typ1]
            n_states2 = self.l_n_states[typ2]
            n_w = n_features * n_states1 * n_states2
            if n_w:
                # n_states1*n_states2 x nb_feat
                pw_typ_typ = wpw[i_w:i_w + n_w].reshape(n_features, -1)
                l_pairwise_potentials.append(np.dot(edge_features,
                                                    pw_typ_typ
                                                    ).reshape(n_edges,
                                                              n_states1,
                                                              n_states2))
            else:
                # first reshaping above complains: "ValueError: total size of
                # new array must be unchanged"
                l_pairwise_potentials.append(np.array([]))
            i_w += n_w

        return l_pairwise_potentials

    def joint_feature(self, x, y):
        """Feature vector associated with instance (x, y).

        Feature representation joint_feature, such that the energy of the
        configuration
        (x, y) and a weight vector w is given by np.dot(w,joint_feature(x, y)).

        Parameters
        ----------
        x : tuple
            Input representation.

        y : list of ndarrays or some tuple (internal use!)
            Either y is a list of a integral ndarrays, giving a complete
            labeling for x.
            Or it is the result of a linear programming relaxation. In this
            case, ``y=(unary_marginals, pariwise_marginals)``.

        Returns
        -------
        p : ndarray, shape (size_joint_feature,)
            Feature vector associated with state (x, y).

        """
        self._check_size_x(x)   # call initialize once!
        l_node_features = self._get_node_features(x)
        l_edges, l_edge_features = (self._get_edges(x),
                                    self._get_edge_features(x))
        l_n_nodes = [len(nf) for nf in self._get_node_features(x)]
        l_n_edges = [len(ef) for ef in self._get_edges(x)]

        if isinstance(y, tuple):
            # y is result of relaxation, tuple of unary and pairwise marginals
            unary_marginals, pw = y

            if isinstance(unary_marginals, list):
                # ad3+ returns a list of unaries, nothing to do here!! :)
                l_unary_marginals = unary_marginals
            else:
                # in case we use someother method (not supported for now
                # actually)
                l_unary_marginals = []
                i, j = 0, 0
                # iteration by type
                for (_n_nodes, _n_states) in zip(l_n_nodes, self.l_n_states):
                    _n_binaries = _n_nodes * _n_states
                    _unary_marginals = unary_marginals[i:i+_n_nodes,
                                                       j:j+_n_states]
                    i += _n_nodes
                    j += _n_states
                    l_unary_marginals.append(_unary_marginals)

            if isinstance(pw, list):
                # ad3+ returns a list of pairwise
                l_pw = pw
            else:
                # until we do better in ad3+ inference, but we cannot I think
                # without touching the learners...
                l_pw = []
                i_start = 0
                for _n_edges, (typ1, typ2) in zip(l_n_edges,
                                                  self._iter_type_pairs()):
                    n = self.l_n_states[typ1] * self.l_n_states[typ2]
                    i_stop = i_start + _n_edges
                    i_state_start = self.a_startindex_by_typ_typ[typ1, typ2]
                    _edge_marginals = pw[i_start:i_stop,
                                         i_state_start:i_state_start+n]
                    i_start = i_stop
                    l_pw.append(_edge_marginals)
        else:
            self._check_size_xy(x, y)
            # make one hot encoding per type
            l_unary_marginals = []
            i_start = 0
            for (_n_nodes,
                 _n_states,
                 typ_start_index) in zip(l_n_nodes,
                                         self.l_n_states,
                                         self._l_type_startindex):
                i_stop = i_start + _n_nodes
                _unary_marginals = np.zeros((_n_nodes, _n_states),
                                            dtype=np.int)
                gx = np.ogrid[:_n_nodes]
                _unary_marginals[gx, y[i_start:i_stop]-typ_start_index] = 1
                l_unary_marginals.append(_unary_marginals)
                i_start = i_stop

            # pairwise
            # same thing, but the type of an edge is a pair of node types
            l_pw = []
            node_offset_by_typ = np.cumsum([0]+[0 if n is None
                                                else n.shape[0] for n in x[0]])
            for _n_edges, (typ1, typ2), edges in zip(l_n_edges,
                                                     self._iter_type_pairs(),
                                                     l_edges):
                _n_states_typ1 = self.l_n_states[typ1]
                _n_states_typ2 = self.l_n_states[typ2]
                _pw = np.zeros((_n_edges, _n_states_typ1 * _n_states_typ2))
                if _n_edges:
                    y1 = y[node_offset_by_typ[typ1] + edges[:, 0]]\
                        - self._l_type_startindex[typ1]
                    y2 = y[node_offset_by_typ[typ2] + edges[:, 1]]\
                        - self._l_type_startindex[typ2]
                    assert (0 <= y1).all() and (y1 <=
                                                self.l_n_states[typ1]).all()
                    assert (0 <= y2).all() and (y2 <=
                                                self.l_n_states[typ2]).all()
                    # set the 1s where they should
                    class_pair_ind = (y2 + _n_states_typ2 * y1)
                    _pw[np.arange(_n_edges), class_pair_ind] = 1
                l_pw.append(_pw)

        # UNARY
        l_unary_acc_ravelled = [np.dot(unary_marginals.T, features).ravel()
                                for (unary_marginals, features)
                                in zip(l_unary_marginals, l_node_features)]
        unaries_acc_ravelled = np.hstack(l_unary_acc_ravelled)

        # PW
        l_pw_ravelled = [np.dot(ef.T, pw).ravel() for (ef, pw)
                         in zip(l_edge_features, l_pw)]
        pairwise_acc_ravelled = np.hstack(l_pw_ravelled)

        joint_feature_vector = np.hstack([unaries_acc_ravelled,
                                          pairwise_acc_ravelled])

        return joint_feature_vector

    def loss_augment_unaries(self, l_unary_potentials, y):
        """
        we do it type-wise
        """
        i_start = 0
        a_y = np.asarray(y)

        for typ, (unary_potentials, class_weight) in enumerate(
                zip(l_unary_potentials, self.l_class_weight)):
            n_y = unary_potentials.shape[0]
            # label 0 must correspond to 1st weight
            y_typ = a_y[i_start:i_start+n_y] - self._l_type_startindex[typ]
            loss_augment_unaries(unary_potentials, y_typ, class_weight)
            i_start += n_y
