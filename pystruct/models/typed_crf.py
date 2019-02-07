# -*- coding: utf-8 -*-

"""
    CRF with different types of nodes

    NOTE: this is an abstract class. Do not use directly.

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

from .crf import CRF
from ..inference import get_installed


class InconsistentLabel(Exception):
    pass


class TypedCRF(CRF):
    """Abstract base class"""
    def __init__(self,
                 n_types,               # how many node type?
                 l_n_states,            # how many labels   per node type?
                 l_n_features,          # how many features per node type?
                 inference_method="ad3",
                 l_class_weight=None):  # class_weight per node type or None
                                        #    <list of array-like> or None

        if inference_method is None:
            # get first in list that is installed
            inference_method = get_installed(['ad3+', 'ad3'])[0]
        self.setInferenceMethod(inference_method)

        self.inference_calls = 0
        # if inference cannot be done, raises an exception
        self.inference_exception = False

        if len(l_n_states) != n_types:
            raise ValueError("Expected 1 number of states per node type.")
        if l_n_features is not None and len(l_n_features) != n_types:
            raise ValueError("Expected 1 number pf features per node type.")
        self.n_types = n_types
        self.l_n_states = l_n_states
        self._n_states = sum(l_n_states)     # total number of states
        self.l_n_features = l_n_features
        self._n_features = sum(self.l_n_features)  # total number of node feat.

        # number of typextype states, or number of states per type of edge
        self.l_n_edge_states = [n1 * n2
                                for n1 in self.l_n_states
                                for n2 in self.l_n_states]

        # class weights:
        # either we get class weights for all types of nodes
        # , or for none of them!
        if l_class_weight:
            if len(l_class_weight) != self.n_types:
                raise ValueError("Expected 1 class weight list per node type.")
            for i, n_states in enumerate(self.l_n_states):
                if len(l_class_weight[i]) != n_states:
                    raise ValueError("Expected 1 class weight per state"
                                     " per node type. Wrong for type %d" % i)

            # class weights are computed by type and simply concatenated
            self.l_class_weight = [np.asarray(class_weight)
                                   for class_weight in l_class_weight]
        else:
            self.l_class_weight = [np.ones(n) for n in self.l_n_states]
        self.class_weight = np.hstack(self.l_class_weight)

        self._set_size_joint_feature()

        # internal stuff
        # when putting node states in a single sequence, index of 1st state
        #  for type i
        self._l_type_startindex = [sum(self.l_n_states[:i])
                                   for i in range(self.n_types+1)]

        # when putting edge states in a single sequence, index of 1st state of
        #  an edge of type (typ1, typ2)
        self.a_startindex_by_typ_typ = np.zeros((self.n_types, self.n_types),
                                                dtype=np.uint32)
        i_state_start = 0
        for typ1, typ1_n_states in enumerate(self.l_n_states):
            for typ2, typ2_n_states in enumerate(self.l_n_states):
                self.a_startindex_by_typ_typ[typ1, typ2] = i_state_start
                i_state_start += typ1_n_states*typ2_n_states

    # -------------- CONVENIENCE --------------------------
    def setInferenceMethod(self, inference_method):
        if inference_method in ["ad3", "ad3+"]:
            self.inference_method = inference_method
        else:
            raise Exception("You must use ad3 or ad3+ as inference method")

    def flattenY(self, lY_by_typ):
        """
        It is more convenient to have the Ys grouped by type, as the Xs are,
        and to have the first label of each type encoded as 0.

        This method does the job. It returns a flat Y array, with unique code
        per class label, which can be passed to 'fit'
        """
        lY = list()
        for n_start_state, Y_typ in zip(self._l_type_startindex, lY_by_typ):
            lY.append(np.asarray(Y_typ) + n_start_state)
        return np.hstack(lY)

    def unflattenY(self, X, flatY):
        """
        predict returns a flat array of Y (same structure as for 'fit')
        This method structures the Y as a list of Y_per_type, where the first
        label of any type is 0
        """
        lY = list()
        i_start_node = 0
        (l_node_features, l_edges, l_edge_features) = X
        for n_start_state, nf in zip(self._l_type_startindex, l_node_features):
            n_nodes = nf.shape[0]
            Y = flatY[i_start_node: i_start_node+n_nodes] - n_start_state
            lY.append(Y)
            i_start_node += n_nodes
        if flatY.shape != (i_start_node,):
            raise ValueError("The total number of label does not match the"
                             " total number of nodes:"
                             " %d != %d" % (flatY.shape[0], i_start_node))
        return lY

    def initialize(self, X, Y=None):
        """
        It is optional to call it. Does data checking only!
        """
        if isinstance(X, list):
            map(self._check_size_x, X)
            if not (Y is None):
                map(self._check_size_xy, X, Y)
        else:
            self._check_size_x(X)
            self._check_size_xy(X, Y)

    def setInferenceException(self, bRaiseExceptionWhenInferenceNotSuccessful):
        """
        set exception on or off when inference canoot be done.
        """
        self.inference_exception = bRaiseExceptionWhenInferenceNotSuccessful
        return self.inference_exception

    # -------------- INTERNAL STUFF --------------------------
    def _set_size_joint_feature(self):
        """
        We have:
        - 1 weight per node feature per label per node type
        """
        self.size_unaries = sum(n_states * n_features for n_states, n_features
                                in zip(self.l_n_states, self.l_n_features)
                                )
        self.size_joint_feature = self.size_unaries

    def __repr__(self):
        return ("%s(n_states: %s, inference_method: %s)"
                % (type(self).__name__, self.l_n_states,
                   self.inference_method))

    def _check_size_x(self, x):
        # node_features are [  i_in_typ -> features ]
        l_node_features = self._get_node_features(x)
        if len(l_node_features) != self.n_types:
            raise ValueError("Expected one node feature array per node type.")

        for typ, typ_features in enumerate(l_node_features):
            if typ_features.shape[1] != self.l_n_features[typ]:
                raise ValueError("Expected %d features for type"
                                 " %d" % (self.l_n_features[typ], typ))

        # edges
        l_edges = self._get_edges(x)
        for edges in l_edges:
            if edges is None:
                continue
            if edges.ndim != 2:
                raise ValueError("Expected a 2 dimensions edge arrays")
            if edges.shape[1] != 2:
                raise ValueError("Expected 2 columns in edge arrays")

        for typ1, typ2 in self._iter_type_pairs():
            edges = self._get_edges_by_type(x, typ1, typ2)

            if edges is None or len(edges) == 0:
                continue
            # edges should point to valid node indices
            nodes1, nodes2 = edges[:, 0], edges[:, 1]
            if min(nodes1) < 0 or min(nodes2) < 0:
                raise ValueError("At least one edge points to negative and"
                                 " therefore invalid node index:"
                                 " type %d to type %d" % (typ1, typ2))
            if max(nodes1) >= l_node_features[typ1].shape[0]:
                raise ValueError("At least one edge starts from a non-existing"
                                 " node index:"
                                 " type %d to type %d" % (typ1, typ2))
            if max(nodes2) >= l_node_features[typ2].shape[0]:
                raise ValueError("At least one edge points to a non-existing"
                                 " node index:"
                                 " type %d to type %d" % (typ1, typ2))
        return True

    def _check_size_xy(self, X, Y):
        if Y is None:
            return

        # make sure Y has the proper length and acceptable labels
        l_node_features = self._get_node_features(X)

        nb_nodes = sum(nf.shape[0] for nf in l_node_features)
        if Y.shape[0] != nb_nodes:
            raise ValueError("Expected 1 label for each of the %d nodes. Got"
                             " %d labels." % (nb_nodes, Y.shape[0]))

        i_start = 0
        for typ, nf, n_states in zip(range(self.n_types),
                                     l_node_features,
                                     self.l_n_states):
            nb_nodes = nf.shape[0]
            if nb_nodes == 0:
                continue
            Y_typ = Y[i_start:i_start+nb_nodes]
            if np.min(Y_typ) < 0:
                raise ValueError("Got a negative label for type %d" % typ)
            if np.min(Y_typ) < self._l_type_startindex[typ]:
                raise InconsistentLabel("labels of type %d start at %d"
                                        "" % (typ,
                                              self._l_type_startindex[typ]))
            if np.max(Y_typ) >= self._l_type_startindex[typ+1]:
                raise InconsistentLabel("labels of type %d end at %d"
                                        "" % (typ,
                                              self._l_type_startindex[typ+1]-1)
                                        )
            i_start = i_start + nb_nodes
        return True

    def _get_node_features(self, x):
        # we replace None by empty array with proper shape
        return [np.empty((0, _n_feat)) if node_features is None
                else node_features
                for (node_features, _n_feat) in zip(x[0], self.l_n_features)]

    def _get_edges(self, x):
        return [np.empty((0, 2)) if edges is None or len(edges) == 0
                else edges for edges in x[1]]

    def _get_edges_by_type(self, x, typ1, typ2):
        return x[1][typ1 * self.n_types+typ2]

    def _iter_type_pairs(self):
        for typ1 in range(self.n_types):
            for typ2 in range(self.n_types):
                yield (typ1, typ2)
        return

    def _get_unary_potentials(self, x, w):
        """Computes unary potentials for x and w.

        Parameters
        ----------
        x : tuple
            Instance Representation.

        w : ndarray, shape=(size_joint_feature,)
            Weight vector for CRF instance.

        Returns
        -------
        unaries : list of ndarray, shape=( n_nodes_typ, n_states_typ )
            Unary weights.
        """
        self._check_size_w(w)
        l_node_features = self._get_node_features(x)

        l_unary_potentials = []

        i_w = 0
        for (features, n_states, n_features) in zip(l_node_features,
                                                    self.l_n_states,
                                                    self.l_n_features):
            n_w = n_states*n_features
            l_unary_potentials.append(
                np.dot(features,
                       w[i_w:i_w+n_w].reshape(n_states,
                                              n_features).T
                       )
                                      )
            i_w += n_w
        assert i_w == self.size_unaries

        # nodes x features  .  features x states  -->  nodes x states
        return l_unary_potentials

    def continuous_loss(self, y, l_y_hat):
        # continuous version of the loss
        # y is the result of linear programming
        # BUT, in multitype mode, y_hat is a list of unaries
        l_result = list()
        cum_n_node = 0
        cum_n_state = 0
        for y_hat in l_y_hat:
            n_node, n_state = y_hat.shape
            # all entries minus correct ones
            # select the correct range of labels and make the labels start at 0
            y_type = y[cum_n_node:cum_n_node+n_node] - cum_n_state
            gx = np.indices(y_type.shape)
            result = 1 - y_hat[gx, y_type]
            l_result.append(result)
            cum_n_node += n_node
            cum_n_state += n_state
        result = np.hstack(l_result)

        if hasattr(self, 'class_weight'):
            return np.sum(self.class_weight[y] * result)
        return np.sum(result)
