import numpy as np
from crf import StructuredProblem
from pyqpbo import alpha_expansion_graph

from IPython.core.debugger import Tracer
tracer = Tracer()


class LatentFixedGraphCRF(StructuredProblem):
    """CRF with general graph that is THE SAME for all examples.
    graph is given by scipy sparse adjacency matrix.
    """
    def __init__(self, n_labels, n_states_per_label, graph):
        self.n_labels = n_labels
        self.n_states_per_label = n_states_per_label
        # n_labels unary parameters, upper triangular for pairwise
        n_states = n_labels * n_states_per_label
        self.n_states = n_states
        self.size_psi = n_states + n_states * (n_states + 1) / 2
        self.graph = graph
        self.edges = np.c_[graph.nonzero()].copy("C")

    def psi(self, x, h, y):
        # x is unaries
        # h is latent labeling
        # y is a labeling
        ## unary features:
        gx = np.ogrid[:x.shape[0]]
        selected_unaries = x[gx, y]
        unaries_acc = np.bincount(h.ravel(), selected_unaries.ravel(),
                minlength=self.n_states)

        ##accumulated pairwise
        #make one hot encoding
        states = np.zeros((y.shape[0], self.n_states),
                dtype=np.int)
        gx = np.ogrid[:y.shape[0]]
        states[gx, h] = 1

        neighbors = self.graph * states
        pw = np.dot(neighbors.T, states)

        feature = np.hstack([unaries_acc, pw[np.tri(self.n_states,
            dtype=np.bool)]])
        return feature

    def loss(self, y, y_hat):
        # hamming loss:
        return np.sum(y != y_hat)

    def inference(self, x, w):
        # augment unary potentials for latent states
        x_wide = np.repeat(x, self.n_states_per_label, axis=1)
        # do usual inference
        unary_params = w[:self.n_states]
        pairwise_flat = np.asarray(w[self.n_states:])
        pairwise_params = np.zeros((self.n_states, self.n_states))
        pairwise_params[np.tri(self.n_states, dtype=np.bool)] = pairwise_flat
        pairwise_params = pairwise_params + pairwise_params.T\
                - np.diag(np.diag(pairwise_params))
        unaries = (- 10 * unary_params * x_wide).astype(np.int32)
        pairwise = (-10 * pairwise_params).astype(np.int32)
        h = alpha_expansion_graph(self.edges, unaries, pairwise)
        # create y from h:
        y = h / self.n_states_per_label
        return h, y

    def latent(self, x, y, w):
        # augment unary potentials for latent states
        x_wide = np.repeat(x, self.n_states_per_label, axis=1)
        # do usual inference
        unary_params = w[:self.n_states]
        pairwise_flat = np.asarray(w[self.n_states:])
        pairwise_params = np.zeros((self.n_states, self.n_states))
        pairwise_params[np.tri(self.n_states, dtype=np.bool)] = pairwise_flat
        pairwise_params = pairwise_params + pairwise_params.T\
                - np.diag(np.diag(pairwise_params))
        unaries = (- 10 * unary_params * x_wide).astype(np.int32)
        # forbid h that is incompoatible with y
        # by modifying unary params
        other_states = (np.arange(self.n_states) / self.n_states_per_label !=
                y[:, np.newaxis])
        unaries[other_states] = +10000
        pairwise = (-10 * pairwise_params).astype(np.int32)
        h = alpha_expansion_graph(self.edges, unaries, pairwise)
        if (h / self.n_states_per_label != y).any():
            if np.any(w):
                print("inconsistent h and y")
                tracer()
            else:
                h = y * self.n_states_per_label
        return h
