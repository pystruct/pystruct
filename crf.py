import numpy as np

from pyqpbo import binary_grid
from pyqpbo import alpha_expansion_grid, alpha_expansion_graph


from IPython.core.debugger import Tracer
tracer = Tracer()


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

    def loss_augmented_inference(self, x, y, w):
        print("FALLBACK no loss augmented inference found")
        return self.inference(x, w)


class BinaryGridCRF(StructuredProblem):
    def __init__(self):
        super(BinaryGridCRF, self).__init__()
        self.n_states = 2
        # one parameter for binary, one for unaries
        self.size_psi = 2

    def psi(self, x, y):
        # x is unaries
        # y is a labeling
        ## unary features:
        gx, gy = np.ogrid[:x.shape[0], :x.shape[1]]
        unaries_acc = np.sum(x[gx, gy, y])

        ##accumulated pairwise
        #make one hot encoding
        labels = np.zeros((y.shape[0], y.shape[1], self.n_states),
                dtype=np.int)
        gx, gy = np.ogrid[:y.shape[0], :y.shape[1]]
        labels[gx, gy, y] = 1
        # vertical edges
        vert = np.dot(labels[1:, :, :].reshape(-1, 2).T, labels[:-1, :,
           :].reshape(-1, 2))
        # horizontal edges
        horz = np.dot(labels[:, 1:, :].reshape(-1, 2).T, labels[:, :-1,
           :].reshape(-1, 2))
        pw = vert + horz
        pw[0, 1] += pw[1, 0]
        #pw = np.zeros((2, 2))
        return np.array([unaries_acc, pw[0, 1]])

    def inference(self, x, w):
        if w.shape != (self.size_psi,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                    (self.size_psi, w.shape))
        self.inference_calls += 1
        unary_param = w[0]
        pairwise_params = np.array([[0, w[1]], [w[1], 0]])
        if (x[:, :, 1] != 0).any():
            raise ValueError("For simplicity, in binary CRFS,"
                    "all entries in the second feature should be 0.")

        unaries = - 1000 * unary_param * x.copy()
        pairwise = -1000 * pairwise_params
        y = binary_grid(unaries.astype(np.int32), pairwise.astype(np.int32))
        return y

    def loss_augmented_inference(self, x, y, w):
        if w.shape != (self.size_psi,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                    (self.size_psi, w.shape))
        unary_param = w[0]
        if unary_param == 0:
            # avoid division by zero
            unary_param = 1e-10
        # do loss augmentation
        gx, gy = np.ogrid[:x.shape[0], :x.shape[1]]
        x_ = x.copy()
        x_[y == 0, 0] -= 1. / unary_param
        x_[y != 0, 0] += 1. / unary_param
        return self.inference(x_, w)


class MultinomialGridCRF(StructuredProblem):
    def __init__(self, n_states=2):
        super(MultinomialGridCRF, self).__init__()
        self.n_states = n_states
        # n_states unary parameters, upper triangular for pairwise
        self.size_psi = n_states + n_states * (n_states + 1) / 2

    def psi(self, x, y):
        # x is unaries
        # y is a labeling
        ## unary features:
        gx, gy = np.ogrid[:x.shape[0], :x.shape[1]]
        selected_unaries = x[gx, gy, y]
        unaries_acc = np.sum(x[gx, gy, y])
        unaries_acc = np.bincount(y.ravel(), selected_unaries.ravel(),
                minlength=self.n_states)

        ##accumulated pairwise
        #make one hot encoding
        labels = np.zeros((y.shape[0], y.shape[1], self.n_states),
                dtype=np.int)
        gx, gy = np.ogrid[:y.shape[0], :y.shape[1]]
        labels[gx, gy, y] = 1
        # vertical edges
        vert = np.dot(labels[1:, :, :].reshape(-1, self.n_states).T,
                labels[:-1, :, :].reshape(-1, self.n_states))
        # horizontal edges
        horz = np.dot(labels[:, 1:, :].reshape(-1, self.n_states).T, labels[:,
            :-1, :].reshape(-1, self.n_states))
        pw = vert + horz
        pw = pw + pw.T - np.diag(np.diag(pw))
        feature = np.hstack([unaries_acc, pw[np.tri(self.n_states,
            dtype=np.bool)]])
        return feature

    def inference(self, x, w):
        self.inference_calls += 1
        if w.shape != (self.size_psi,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                    (self.size_psi, w.shape))
        unary_params = w[:self.n_states]
        pairwise_flat = np.asarray(w[self.n_states:])
        pairwise_params = np.zeros((self.n_states, self.n_states))
        pairwise_params[np.tri(self.n_states, dtype=np.bool)] = pairwise_flat
        pairwise_params = pairwise_params + pairwise_params.T\
                - np.diag(np.diag(pairwise_params))
        unaries = (-10000 * unary_params * x).astype(np.int32)
        pairwise = (-10000 * pairwise_params).astype(np.int32)
        y = alpha_expansion_grid(unaries, pairwise)
        return y

    def loss_augmented_inference(self, x, y, w):
        if w.shape != (self.size_psi,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                    (self.size_psi, w.shape))
        unary_params = w[:self.n_states]
        # avoid division by zero:
        unary_params[unary_params == 0] = 1e-10
        x_ = x.copy()
        for l in np.arange(self.n_states):
            # for each class, decrement unaries
            # for loss-agumention
            x_[y != l, l] += 1. / unary_params[l]
        return self.inference(x_, w)


class MultinomialFixedGraphCRF(StructuredProblem):
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

        feature = np.hstack([unaries_acc, pw[np.tri(self.n_states,
            dtype=np.bool)]])
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
        pairwise_params = pairwise_params + pairwise_params.T\
                - np.diag(np.diag(pairwise_params))
        unaries = (-1000 * unary_params * x).astype(np.int32)
        pairwise = (-1000 * pairwise_params).astype(np.int32)
        y = alpha_expansion_graph(self.edges, unaries, pairwise, random_seed=1)
        return y

    def loss_augmented_inference(self, x, y, w):
        if w.shape != (self.size_psi,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                    (self.size_psi, w.shape))
        unary_params = w[:self.n_states].copy()
        # avoid division by zero:
        unary_params[unary_params == 0] = 1e-10
        x_ = x.copy()
        for l in np.arange(self.n_states):
            # for each class, decrement unaries
            # for loss-agumention
            x_[y != l, l] += 1. / unary_params[l]
        return self.inference(x_, w)


class MultinomialFixedGraphCRFNoBias(MultinomialGridCRF):
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
        pairwise_params[np.tri(self.n_states, k=-1, dtype=np.bool)] = \
                pairwise_flat
        pairwise_params = pairwise_params + pairwise_params.T\
                - np.diag(np.diag(pairwise_params))
        unaries = (-1000 * unary * x).astype(np.int32)
        pairwise = (-1000 * pairwise_params).astype(np.int32)
        y = alpha_expansion_graph(self.edges, unaries, pairwise, random_seed=1)
        from pyqpbo import binary_graph
        y_ = binary_graph(self.edges, unaries, pairwise)
        if (y_ != y).any():
            tracer()
        return y

    def loss_augmented_inference(self, x, y, w):
        if w.shape != (self.size_psi,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                    (self.size_psi, w.shape))
        x_ = x.copy()
        for l in np.arange(self.n_states):
            # for each class, decrement unaries
            # for loss-agumention
            x_[y != l, l] += 1.
        return self.inference(x_, w)
