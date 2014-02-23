import numpy as np

from . import GridCRF, DirectionalGridCRF
from .latent_graph_crf import kmeans_init, LatentGraphCRF
from ..utils import make_grid_edges


class LatentGridCRF(GridCRF, LatentGraphCRF):
    """Latent variable CRF with 2d grid graph.
    """
    def __init__(self, n_labels=None, n_features=None, n_states_per_label=2,
                 inference_method=None, neighborhood=4):
        LatentGraphCRF.__init__(self, n_labels, n_features, n_states_per_label,
                                inference_method=inference_method)
        GridCRF.__init__(self, n_states=self.n_states,
                         n_features=self.n_features, neighborhood=neighborhood,
                         inference_method=inference_method)

    def _set_size_joint_feature(self):
        LatentGraphCRF._set_size_joint_feature(self)

    def initialize(self, X, Y):
        LatentGraphCRF.initialize(self, X, Y)

    def init_latent(self, X, Y):
        # treat all edges the same
        edges = [[make_grid_edges(x, neighborhood=self.neighborhood,
                                  return_lists=False)] for x in X]
        H = kmeans_init(X.reshape(X.shape[0], -1, self.n_features),
                        Y.reshape(Y.shape[0], -1), edges,
                        n_labels=self.n_labels,
                        n_states_per_label=self.n_states_per_label)
        return np.array(H).reshape(Y.shape)

    def loss_augmented_inference(self, x, h, w, relaxed=False,
                                 return_energy=False):
        h = LatentGraphCRF.loss_augmented_inference(self, x, h.ravel(), w,
                                                    relaxed, return_energy)
        return self._reshape_y(h, x.shape, return_energy)

    def latent(self, x, y, w):
        res = LatentGraphCRF.latent(self, x, y.ravel(), w)
        return res.reshape(y.shape)

    def continuous_loss(self, y, y_hat):
        # continuous version of the loss
        # y_hat is the result of linear programming
        return LatentGraphCRF.continuous_loss(
            self, y.ravel(), y_hat.reshape(-1, y_hat.shape[-1]))


class LatentDirectionalGridCRF(DirectionalGridCRF, LatentGridCRF):
    """Latent variable CRF with directional 2d grid graph.

    Multiple inheritance weirdness....
    All features / potentials are the same as in the DirectionalGridCRF,
    so use these.

    Things that have to do with y or h need to call the
    LatentGridCRF function - that simply works because the feature are right.
    """
    def __init__(self, n_labels=None, n_features=None, n_states_per_label=2,
                 inference_method=None, neighborhood=4):
        self.neighborhood = neighborhood
        self.symmetric_edge_features = []
        self.antisymmetric_edge_features = []
        self.n_edge_features = 2 if neighborhood == 4 else 4
        LatentGridCRF.__init__(self, n_labels, n_features, n_states_per_label,
                               inference_method=inference_method)

    def _set_size_joint_feature(self):
        LatentGridCRF._set_size_joint_feature(self)
        DirectionalGridCRF._set_size_joint_feature(self)

    def initialize(self, X, Y):
        LatentGridCRF.initialize(self, X, Y)

    def init_latent(self, X, Y):
        edges = [make_grid_edges(x, neighborhood=self.neighborhood,
                                 return_lists=True) for x in X]
        H = kmeans_init(X.reshape(X.shape[0], -1, self.n_features),
                        Y.reshape(Y.shape[0], -1), edges,
                        n_labels=self.n_labels,
                        n_states_per_label=self.n_states_per_label,
                        symmetric=False)
        return np.array(H).reshape(Y.shape)

    def loss_augmented_inference(self, x, h, w, relaxed=False):
        h = LatentGridCRF.loss_augmented_inference(self, x, h, w,
                                                   relaxed=relaxed)
        return h
