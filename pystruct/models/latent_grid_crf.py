
from . import GridCRF, DirectionalGridCRF
from .latent_graph_crf import LatentGraphCRF


class LatentGridCRF(GridCRF, LatentGraphCRF):
    """Latent variable CRF with 2d grid graph.

    This is also called "hidden dynamics CRF".
    For each output variable there is an additional variable which
    can encode additional states and interactions.

    The input is the same as for GridCRF.

    Parameters
    ----------
    n_labels : int or None, default=None
        Number of states of output variables.
        Inferred from the data if None.

    n_featues : int or None (default=None).
        Number of input features per input variable.
        ``None`` means it is inferred from data.

    n_states_per_label : int or list (default=2)
        Number of latent states associated with each observable state.
        Can be either an integer, which means the same number
        of hidden states will be used for all observable states, or a list
        of integers of length ``n_labels``.

    inference_method : string, default="ad3"
        Function to call to do inference and loss-augmented inference.
        Possible values are:

            - 'max-product' for max-product belief propagation.
                Recommended for chains an trees. Loopy belief propagatin in case of a general graph.
            - 'lp' for Linear Programming relaxation using cvxopt.
            - 'ad3' for AD3 dual decomposition.
            - 'qpbo' for QPBO + alpha expansion.
            - 'ogm' for OpenGM inference algorithms.

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
        H = LatentGraphCRF.init_latent(self, X, Y)
        return [h.reshape(y.shape) for h, y in zip(H, Y)]

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

    This is also called "hidden dynamics CRF".
    For each output variable there is an additional variable which
    can encode additional states and interactions.

    The input is the same as for GridCRF, directional behavior as
    in DirectionalGridCRF.

    Parameters
    ----------
    n_labels : int or None, default=None
        Number of states of output variables.
        Inferred from the data if None.

    n_featues : int or None (default=None).
        Number of input features per input variable.
        Inferred from the data if None.

    n_states_per_label : int or list (default=2)
        Number of latent states associated with each observable state.
        Can be either an integer, which means the same number
        of hidden states will be used for all observable states, or a list
        of integers of length ``n_labels``.

    inference_method : string, default="ad3"
        Function to call to do inference and loss-augmented inference.
        Possible values are:

            - 'max-product' for max-product belief propagation.
                Recommended for chains an trees. Loopy belief propagatin in case of a general graph.
            - 'lp' for Linear Programming relaxation using cvxopt.
            - 'ad3' for AD3 dual decomposition.
            - 'qpbo' for QPBO + alpha expansion.
            - 'ogm' for OpenGM inference algorithms.

    """
    #Multiple inheritance weirdness....
    #All features / potentials are the same as in the DirectionalGridCRF,
    #so use these.

    #Things that have to do with y or h need to call the
    #LatentGridCRF function - that simply works because the feature are right.
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

    def loss_augmented_inference(self, x, h, w, relaxed=False):
        h = LatentGridCRF.loss_augmented_inference(self, x, h, w,
                                                   relaxed=relaxed)
        return h
