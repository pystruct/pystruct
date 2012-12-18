import itertools

import numpy as np
from pyqpbo import alpha_expansion_graph

from inference_methods import (_inference_qpbo, _inference_dai, _inference_lp,
                               _inference_ad3, _make_grid_edges)

from IPython.core.debugger import Tracer
tracer = Tracer()


def pairwise_grid_features(grid_labels, neighborhood=4):
    if neighborhood not in [4, 8]:
        raise ValueError("neighborhood has to be 4 or 8.")
    n_states = grid_labels.shape[-1]
    features = []
    # horizontal edges
    right = np.dot(grid_labels[:, 1:, :].reshape(-1, n_states).T,
                   grid_labels[:, :-1, :].reshape(-1, n_states))
    features.append(right)
    # vertical edges
    down = np.dot(grid_labels[1:, :, :].reshape(-1, n_states).T,
                  grid_labels[:-1, :, :].reshape(-1, n_states))
    features.append(down)
    if neighborhood == 8:
        upright = np.dot(grid_labels[1:, :-1, :].reshape(-1, n_states).T,
                         grid_labels[:-1, 1:, :].reshape(-1, n_states))
        features.append(upright)
        downright = np.dot(grid_labels[:-1, :-1, :].reshape(-1, n_states).T,
                           grid_labels[1:, 1:, :].reshape(-1, n_states))
        features.append(downright)
    return features


def unwrap_pairwise(y):
    """given a y that may contain pairwise marginals, yield plain y."""
    if isinstance(y, tuple):
        return y[0]
    return y


class StructuredProblem(object):
    def __init__(self):
        self.size_psi = None
        self.inference_calls = 0

    def psi(self, x, y):
        pass

    def inference(self, x, w):
        pass

    def loss(self, y, y_hat):
        # hamming loss:
        return np.sum(y != y_hat)

    def continuous_loss(self, y, y_hat):
        # continuous version of the loss
        # y is the result of linear programming
        y_one_hot = np.zeros_like(y_hat)
        gx, gy = np.indices(y.shape)
        y_one_hot[gx, gy, y] = 1
        # all entries minus correct ones
        return np.prod(y.shape) - np.sum(y_one_hot * y_hat)

    def loss_augmented_inference(self, x, y, w):
        print("FALLBACK no loss augmented inference found")
        return self.inference(x, w)


class CRF(StructuredProblem):
    def __repr__(self):
        return ("GridCRF, n_states: %d, inference_method: %s"
                % (self.n_states, self.inference_method))

    def loss_augment(self, x, y, w):
        unary_params = w[:self.n_states].copy()
        # avoid division by zero:
        if (unary_params == 0).any():
            raise ValueError("Unary params are exactly zero, can not do"
                             " loss-augmentation!")
        x_ = x.copy()
        for l in np.arange(self.n_states):
            # for each class, decrement unaries
            # for loss-agumention
            x_[y != l, l] += 1. / unary_params[l]
        return x_

    def loss_augmented_inference(self, x, y, w, relaxed=False):
        if w.shape != (self.size_psi,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             (self.size_psi, w.shape))
        x_ = self.loss_augment(x, y, w)
        return self.inference(x_, w, relaxed)


class GridCRF(CRF):
    """Pairwise CRF on a 2d grid.

    Pairwise potentials are symmetric and the same for all edges.
    This leads to n_classes parameters for unary potentials and
    n_classes * (n_classes + 1) / 2 parameters for edge potentials.

    Parameters
    ----------
    n_states : int, default=2
        Number of states for all variables.

    inference_method : string, default="qpbo"
        Function to call do do inference and loss-augmented inference.
        Possible values are:

            - 'qpbo' for QPBO + alpha expansion.
            - 'dai' for LibDAI bindings (which has another parameter).
            - 'lp' for Linear Programming relaxation using GLPK.
            - 'ad3' for AD3 dual decomposition.

    neighborhood: int, default=4
        Neighborhood defining connection for each variable in the grid.
        Possible choices are 4 and 8.
    """
    def __init__(self, n_states=2, inference_method='qpbo', neighborhood=4):
        super(GridCRF, self).__init__()
        self.neighborhood = neighborhood
        self.n_states = n_states
        self.inference_method = inference_method
        # n_states unary parameters, upper triangular for pairwise
        self.size_psi = n_states + n_states * (n_states + 1) / 2

    def get_pairwise_weights(self, w):
        if w.shape != (self.size_psi,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             (self.size_psi, w.shape))
        pairwise_flat = np.asarray(w[self.n_states:])
        pairwise_params = np.zeros((self.n_states, self.n_states))
        # set lower triangle of matrix, then make symmetric
        # we could try to redo this using ``scipy.spatial.distance`` somehow
        pairwise_params[np.tri(self.n_states, dtype=np.bool)] = pairwise_flat
        return (pairwise_params + pairwise_params.T -
                np.diag(np.diag(pairwise_params)))

    def get_unary_weights(self, w):
        if w.shape != (self.size_psi,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             (self.size_psi, w.shape))
        return w[:self.n_states]

    def psi(self, x, y):
        # x is unaries
        # y is a labeling
        if isinstance(y, tuple):
            # y can also be continuous (from lp)
            # in this case, it comes with accumulated edge marginals
            y, pw = y
            x_flat = x.reshape(-1, x.shape[-1])
            y_flat = y.reshape(-1, y.shape[-1])
            unaries_acc = np.sum(x_flat * y_flat, axis=0)
            labels = y
        else:
            ## unary features:
            gx, gy = np.ogrid[:x.shape[0], :x.shape[1]]
            selected_unaries = x[gx, gy, y]
            unaries_acc = np.bincount(y.ravel(), selected_unaries.ravel(),
                                      minlength=self.n_states)

            ##accumulated pairwise
            #make one hot encoding
            labels = np.zeros((y.shape[0], y.shape[1], self.n_states),
                              dtype=np.int)
            labels[gx, gy, y] = 1
            pw = np.sum(pairwise_grid_features(labels, self.neighborhood),
                        axis=0)

        pw = pw + pw.T - np.diag(np.diag(pw))
        feature = np.hstack([unaries_acc,
                             pw[np.tri(self.n_states, dtype=np.bool)]])
        return feature

    def inference(self, x, w, relaxed=False):
        unary_params = self.get_unary_weights(w)
        pairwise_params = self.get_pairwise_weights(w)
        self.inference_calls += 1
        edges = _make_grid_edges(x, neighborhood=self.neighborhood)
        if self.inference_method == "qpbo":
            return _inference_qpbo(x, unary_params, pairwise_params,
                                   edges)
        elif self.inference_method == "dai":
            return _inference_dai(x, unary_params, pairwise_params,
                                  edges)
        elif self.inference_method == "lp":
            return _inference_lp(x, unary_params, pairwise_params,
                                 edges, relaxed)
        elif self.inference_method == "ad3":
            return _inference_ad3(x, unary_params, pairwise_params,
                                  edges, relaxed)
        else:
            raise ValueError("inference_method must be 'qpbo' or 'dai', got %s"
                             % self.inference_method)


class DirectionalGridCRF(CRF):
    """CRF with different kind of edges, each having their own parameters.

    Pairwise potentials are not symmetric and are independend for each kind of
    edges.  This leads to n_classes parameters for unary potentials and
    n_edge_types * n_classes ** 2 parameters for edge potentials.
    The number of edge-types is two for a 4-connected neighborhood
    (horizontal and vertical) or 4 for a 8 connected neighborhood (additionally
    two diagonals).

    Parameters
    ----------
    n_states : int, default=2
        Number of states for all variables.

    inference_method : string, default="qpbo"
        Function to call do do inference and loss-augmented inference.
        Possible values are:

            - 'qpbo' for QPBO + alpha expansion.
            - 'dai' for LibDAI bindings (which has another parameter).
            - 'lp' for Linear Programming relaxation using GLPK.
            - 'ad3' for AD3 dual decomposition.

    neighborhood : int, default=4
        Neighborhood defining connection for each variable in the grid.
        Possible choices are 4 and 8.
    """
    def __init__(self, n_states=2, inference_method='lp', neighborhood=4):
        super(DirectionalGridCRF, self).__init__()
        self.n_states = n_states
        self.inference_method = inference_method
        self.n_edge_types = 2 if neighborhood == 4 else 4
        # n_states unary parameters, upper triangular for pairwise
        self.size_psi = n_states + n_states * n_states * self.n_edge_types
        self.neighborhood = neighborhood

    def get_pairwise_weights(self, w):
        if w.shape != (self.size_psi,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             (self.size_psi, w.shape))
        return w[self.n_states:].reshape(self.n_edge_types, self.n_states,
                                         self.n_states)

    def get_unary_weights(self, w):
        if w.shape != (self.size_psi,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             (self.size_psi, w.shape))
        return w[:self.n_states]

    # TODO
    def psi(self, x, y):
        # x is unaries
        # y is a labeling
        if isinstance(y, tuple):
            # y can also be continuous (from lp)
            # in this case, it comes with accumulated edge marginals
            y, pw = y
            x_flat = x.reshape(-1, x.shape[-1])
            y_flat = y.reshape(-1, y.shape[-1])
            unaries_acc = np.sum(x_flat * y_flat, axis=0)
            labels = y
        else:
            ## unary features:
            gx, gy = np.ogrid[:x.shape[0], :x.shape[1]]
            selected_unaries = x[gx, gy, y]
            unaries_acc = np.bincount(y.ravel(), selected_unaries.ravel(),
                                      minlength=self.n_states)

            ##accumulated pairwise
            #make one hot encoding
            labels = np.zeros((y.shape[0], y.shape[1], self.n_states),
                              dtype=np.int)
            labels[gx, gy, y] = 1
            pw = np.vstack(pairwise_grid_features(labels, self.neighborhood))

        feature = np.hstack([unaries_acc, pw.ravel()])
        return feature

    def inference(self, x, w, relaxed=False):
        self.inference_calls += 1
        # extract unary weights
        unary_params = self.get_unary_weights(w)
        # extract pairwise weights of shape n_edge_types x n_states x n_states
        pairwise_params = self.get_pairwise_weights(w)
        edges = _make_grid_edges(x, neighborhood=self.neighborhood,
                                 return_lists=True)
        n_edges = [len(e) for e in edges]
        # replicate pairwise weights for edges of certain type
        edge_weights = [np.repeat(pw[np.newaxis, :, :], n, axis=0)
                        for pw, n in zip(pairwise_params, n_edges)]
        edge_weights = np.vstack(edge_weights)
        edges = np.vstack(edges)

        #if self.inference_method == "qpbo":
            #return _inference_qpbo(x, unary_params, pairwise_params,
                                   #self.neighborhood)
        #elif self.inference_method == "dai":
            #return _inference_dai(x, unary_params, pairwise_params,
                                  #self.neighborhood)
        if self.inference_method == "lp":
            return _inference_lp(x, unary_params, edge_weights,
                                 edges, relaxed)
        #elif self.inference_method == "ad3":
            #return _inference_ad3(x, unary_params, pairwise_params,
                                  #self.neighborhood, relaxed)
        else:
            raise ValueError("inference_method must be 'qpbo' or 'dai', got %s"
                             % self.inference_method)


class FixedGraphCRF(CRF):
    """CRF with general graph that is THE SAME for all examples.
    graph is given by scipy sparse adjacency matrix.
    """
    def __init__(self, n_states, graph):
        self.n_states = n_states
        # n_states unary parameters, upper triangular for pairwise
        self.size_psi = n_states + n_states * (n_states + 1) / 2
        self.graph = graph
        self.edges = np.c_[graph.nonzero()].copy("C")

    def psi(self, x, y):
        # x is unaries
        # y is a labeling
        ## unary features:
        n_nodes = y.shape[0]
        gx = np.ogrid[:n_nodes]
        selected_unaries = x[gx, y]
        unaries_acc = np.bincount(y.ravel(), selected_unaries.ravel(),
                                  minlength=self.n_states)

        ##accumulated pairwise
        #make one hot encoding
        labels = np.zeros((n_nodes, self.n_states),
                          dtype=np.int)
        gx = np.ogrid[:n_nodes]
        labels[gx, y] = 1

        neighbors = self.graph * labels
        pw = np.dot(neighbors.T, labels)

        feature = np.hstack([unaries_acc,
                             pw[np.tri(self.n_states, dtype=np.bool)]])
        return feature

    def inference(self, x, w):
        if w.shape != (self.size_psi,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             (self.size_psi, w.shape))
        self.inference_calls += 1
        unary_params = w[:self.n_states]
        pairwise_flat = np.asarray(w[self.n_states:])
        pairwise_params = np.zeros((self.n_states, self.n_states))
        pairwise_params[np.tri(self.n_states, dtype=np.bool)] = pairwise_flat
        pairwise_params = (pairwise_params + pairwise_params.T
                           - np.diag(np.diag(pairwise_params)))
        unaries = (-1000 * unary_params * x).astype(np.int32)
        pairwise = (-1000 * pairwise_params).astype(np.int32)
        y = alpha_expansion_graph(self.edges, unaries, pairwise, random_seed=1)
        return y


class FixedGraphCRFNoBias(GridCRF):
    """CRF with general graph that is THE SAME for all examples.
    graph is given by scipy sparse adjacency matrix.
    """
    def __init__(self, n_states, graph):
        self.inference_calls = 0
        self.n_states = n_states
        #upper triangular for pairwise
        # last one gives weights to unary potentials
        self.size_psi = n_states * (n_states - 1) / 2 + 1
        self.graph = graph
        self.edges = np.c_[graph.nonzero()].copy("C")

    def psi(self, x, y):
        # x is unaries
        # y is a labeling
        ## unary features:
        n_nodes = y.shape[0]
        gx = np.ogrid[:n_nodes]
        unaries_acc = x[gx, y].sum()

        # x is unaries
        # y is a labeling
        n_nodes = y.shape[0]

        ##accumulated pairwise
        #make one hot encoding
        labels = np.zeros((n_nodes, self.n_states), dtype=np.int)
        gx = np.ogrid[:n_nodes]
        labels[gx, y] = 1

        neighbors = self.graph * labels
        pw = np.dot(neighbors.T, labels)

        pairwise = pw[np.tri(self.n_states, k=-1, dtype=np.bool)]
        feature = np.hstack([unaries_acc, pairwise])
        return feature

    def inference(self, x, w):
        if w.shape != (self.size_psi,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             (self.size_psi, w.shape))
        self.inference_calls += 1
        pairwise_flat = np.asarray(w[:-1])
        unary = w[-1]
        pairwise_params = np.zeros((self.n_states, self.n_states))
        upper_tri = np.tri(self.n_states, k=-1, dtype=np.bool)
        pairwise_params[upper_tri] = pairwise_flat
        pairwise_params = (pairwise_params + pairwise_params.T
                           - np.diag(np.diag(pairwise_params)))
        unaries = (-1000 * unary * x).astype(np.int32)
        pairwise = (-1000 * pairwise_params).astype(np.int32)
        y = alpha_expansion_graph(self.edges, unaries, pairwise, random_seed=1)
        from pyqpbo import binary_graph
        y_ = binary_graph(self.edges, unaries, pairwise)
        if (y_ != y).any():
            tracer()
        return y


def exhaustive_loss_augmented_inference(problem, x, y, w):
    size = np.prod(x.shape[:-1])
    best_y = None
    best_energy = np.inf
    for y_hat in itertools.product(range(problem.n_states), repeat=size):
        y_hat = np.array(y_hat).reshape(x.shape[:-1])
        #print("trying %s" % repr(y_hat))
        psi = problem.psi(x, y_hat)
        energy = -problem.loss(y, y_hat) - np.dot(w, psi)
        if energy < best_energy:
            best_energy = energy
            best_y = y_hat
    return best_y
