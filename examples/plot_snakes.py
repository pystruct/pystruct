import numpy as np

from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix

from pystruct.models import DirectionalGridCRF, EdgeFeatureGraphCRF, CRF
from pystruct.learners import OneSlackSSVM
#from pystruct.learners import NSlackSSVM
from pystruct.datasets import load_snakes


class DirectionalGridCRFFeatures(DirectionalGridCRF):
    def __init__(self, n_states=None, n_features=None, inference_method=None,
                 neighborhood=4):
        self.neighborhood = neighborhood
        EdgeFeatureGraphCRF.__init__(self, n_states, n_features,
                                     inference_method=inference_method)

    def initialize(self, X, Y):
        n_edge_features = X[0].shape[-1] * 4
        if self.n_edge_features is None:
            self.n_edge_features = n_edge_features
        elif self.n_edge_features != n_edge_features:
            raise ValueError("Expected %d edge features, got %d"
                             % (self.n_edge_features, n_edge_features))
        CRF.initialize(self, X, Y)

    def _get_edge_features(self, x):
        right, down = self._get_edges(x, flat=False)
        feat = self._get_features(x)
        all_edges = np.vstack([right, down])
        # there are two kind of edges, and both have "from" and "to"
        # resulting in four possible features.
        edge_features = np.zeros((all_edges.shape[0], x.shape[2], 4))
        edge_features[:len(right), :, 0] = feat[right[:, 0]]
        edge_features[:len(right), :, 1] = feat[right[:, 1]]
        edge_features[len(right):, :, 0] = feat[down[:, 0]]
        edge_features[len(right):, :, 1] = feat[down[:, 1]]
        return edge_features.reshape(all_edges.shape[0], -1)



def one_hot_colors(x):
    x = x / 255
    flat = np.dot(x.reshape(-1, 3),  2 **  np.arange(3))
    one_hot = label_binarize(flat, classes=[1, 2, 3, 4, 6])
    return one_hot.reshape(x.shape[0], x.shape[1], 5)


def neighborhood_feature(x):
    """Add a 3x3 neighborhood around each pixel as a feature."""
    # position 3 is background.
    features = np.zeros((x.shape[0], x.shape[1], 5, 9))
    features[:, :, 3, :] = 1
    features[1:, 1:, :, 0] = x[:-1, :-1, :]
    features[:, 1:, :, 1] = x[:, :-1, :]
    features[:-1, 1:, :, 2] = x[1:, :-1, :]
    features[1:, :, :, 3] = x[:-1, :, :]
    features[:-1, :-1, :, 4] = x[1:, 1:, :]
    features[:-1, :, :, 5] = x[1:, :, :]
    features[1:, :-1, :, 6] = x[:-1, 1:, :]
    features[:, :-1, :, 7] = x[:, 1:, :]
    features[:, :, :, 8] = x[:, :, :]
    return features.reshape(x.shape[0], x.shape[1], -1)



def main():
    snakes = load_snakes()
    X_train, Y_train = snakes['X_train'], snakes['Y_train']

    X_train = [one_hot_colors(x) for x in X_train]

    X_train = [neighborhood_feature(x) for x in X_train]

    crf = DirectionalGridCRFFeatures(inference_method='qpbo')
    ssvm = OneSlackSSVM(crf, inference_cache=0, C=.1, verbose=2,
                        show_loss_every=100, inactive_threshold=1e-5,
                        tol=1e-3, switch_to=("ad3", {"branch_and_bound": True}))
    ssvm.fit(X_train, Y_train)
    print(ssvm.score(X_train, Y_train))
    Y_pred = ssvm.predict(X_train)
    y_pred = np.hstack([y.ravel() for y in Y_pred])
    y_train = np.hstack([y.ravel() for y in Y_train])
    print(confusion_matrix(y_train, y_pred))


if __name__ == "__main__":
    main()
    # results directional grid C=0.1 0.795532646048
    # results one-hot grid C=0.1 0.788395453344
    # completely flat C=1 svc 0.767909066878
    # non-one-hot flat: 0.765662172879
    # with directional grid 3x3 features C=0.1: 0.870737509913
    # ad3 refit C=0.1 0.882632831086
    # unary inference: 0.825270948982
    # pairwise feature classe C=0.1: 0.933254031192
    #final primal objective: 62.272486 gap: 24.587289
    # ad3bb C=0.1 :1.0

