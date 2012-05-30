import numpy as np

from pyqpbo import binary_grid


class StructuredProblem(object):
    def __init__(self):
        self.size_psi = None

    def psi(self, x, y):
        pass

    def inference(self, x, w):
        pass

    def loss(self, y, y_hat):
        pass


class BinaryGridCRF(StructuredProblem):
    def __init__(self):
        self.n_labels = 2
        # three parameter for binary, one for unaries
        self.size_psi = 4

    def psi(self, x, y):
        # x is unaries
        # y is a labeling
        ## unary features:
        gx, gy = np.ogrid[:x.shape[0], :x.shape[1]]
        unaries_acc = np.sum(x[gx, gy, y])

        ##accumulated pairwise
        #make one hot encoding
        labels = np.zeros((y.shape[0], y.shape[1], self.n_labels),
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
        return np.array([unaries_acc, pw[0, 0], pw[0, 1], pw[1, 1]])

    def loss(self, y, y_hat):
        # hamming loss:
        return np.sum(y != y_hat)

    def inference(self, x, w):
        unary_param = w[0]
        pairwise_params = np.array([[w[1], w[2]], [w[2], w[3]]])
        unaries = - 10 * unary_param * x
        pairwise = -10 * pairwise_params
        y = binary_grid(unaries.astype(np.int32), pairwise.astype(np.int32))
        return y
