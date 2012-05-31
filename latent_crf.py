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
        # y is a labeling
        ## unary features:
        gx = np.ogrid[:x.shape[0]]
        selected_unaries = x[gx, y]
        unaries_acc = np.bincount(y.ravel(), selected_unaries.ravel(),
                minlength=self.n_labels)
        if (h / self.n_states_per_label != y).any():
            print("inconsistent h and y")
            tracer()
        tracer()

        ##accumulated pairwise
        #make one hot encoding
        states = np.zeros((y.shape[0], self.n_states),
                dtype=np.int)
        gx = np.ogrid[:y.shape[0]]
        states[gx, h] = 1

        neighbors = self.graph * states
        pw = np.dot(neighbors.T, states)

        feature = np.hstack([unaries_acc, pw[np.tri(self.n_labels,
            dtype=np.bool)]])
        return feature

    def loss(self, y, y_hat):
        # hamming loss:
        return np.sum(y != y_hat)

    def inference(self, x, w):
        unary_params = w[:self.n_labels]
        pairwise_flat = np.asarray(w[self.n_labels:])
        pairwise_params = np.zeros((self.n_labels, self.n_labels))
        pairwise_params[np.tri(self.n_labels, dtype=np.bool)] = pairwise_flat
        pairwise_params = pairwise_params + pairwise_params.T\
                - np.diag(np.diag(pairwise_params))
        unaries = (- 10 * unary_params * x).astype(np.int32)
        pairwise = (-10 * pairwise_params).astype(np.int32)
        y = alpha_expansion_graph(self.edges, unaries, pairwise)
        return y

    def latent(self, x, y, w):
        unary_params = w[:self.n_labels]
        pairwise_flat = np.asarray(w[self.n_labels:])
        pairwise_params = np.zeros((self.n_labels, self.n_labels))
        pairwise_params[np.tri(self.n_labels, dtype=np.bool)] = pairwise_flat
        pairwise_params = pairwise_params + pairwise_params.T\
                - np.diag(np.diag(pairwise_params))
        unaries = (- 10 * unary_params * x).astype(np.int32)
        pairwise = (-10 * pairwise_params).astype(np.int32)
        y = alpha_expansion_graph(self.edges, unaries, pairwise)
        return y
