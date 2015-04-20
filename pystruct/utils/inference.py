import itertools
from sklearn.externals.joblib import Parallel, delayed

import numpy as np


def unwrap_pairwise(y):
    """given a y that may contain pairwise marginals, yield plain y."""
    if isinstance(y, tuple):
        return y[0]
    return y


def expand_sym(sym_compressed):
    """Expand compressed symmetric matrix to full square matrix.

    Similar to scipy.spatial.squareform, but also contains the
    diagonal.
    """
    length = sym_compressed.size
    size = int(np.sqrt(2 * length + 0.25) - 1 / 2.)
    sym = np.zeros((size, size))
    sym[np.tri(size, dtype=np.bool)] = sym_compressed
    return (sym + sym.T - np.diag(np.diag(sym)))


def compress_sym(sym_expanded, make_symmetric=True):
    """Compress symmetric matrix to a vector.

    Similar to scipy.spatial.squareform, but also contains the
    diagonal.

    Parameters
    ----------
    sym_expanded : nd-array, shape (size, size)
        Input matrix to compress.

    make_symmetric : bool (default=True)
        Whether to symmetrize the input matrix before compressing.
        It is made symmetric by using
        ``sym_expanded + sym_expanded.T - np.diag(np.diag(sym_expanded))``
        This makes sense if only one of the two entries was non-zero before.


    """
    size = sym_expanded.shape[0]
    if make_symmetric:
        sym_expanded = (sym_expanded + sym_expanded.T -
                        np.diag(np.diag(sym_expanded)))
    return sym_expanded[np.tri(size, dtype=np.bool)]


## global functions for easy parallelization
def find_constraint(model, x, y, w, y_hat=None, relaxed=True,
                    compute_difference=True):
    """Find most violated constraint, or, given y_hat,
    find slack and djoint_feature for this constraing.

    As for finding the most violated constraint, it is enough to compute
    joint_feature(x, y_hat), not djoint_feature, we can optionally skip
    computing joint_feature(x, y) using compute_differences=False
    """

    if y_hat is None:
        y_hat = model.loss_augmented_inference(x, y, w, relaxed=relaxed)
    joint_feature = model.joint_feature
    if getattr(model, 'rescale_C', False):
        delta_joint_feature = -joint_feature(x, y_hat, y)
    else:
        delta_joint_feature = -joint_feature(x, y_hat)
    if compute_difference:
        if getattr(model, 'rescale_C', False):
            delta_joint_feature += joint_feature(x, y, y)
        else:
            delta_joint_feature += joint_feature(x, y)

    if isinstance(y_hat, tuple):
        # continuous label
        loss = model.continuous_loss(y, y_hat[0])
    else:
        loss = model.loss(y, y_hat)
    slack = max(loss - np.dot(w, delta_joint_feature), 0)
    return y_hat, delta_joint_feature, slack, loss


def find_constraint_latent(model, x, y, w, relaxed=True):
    """Find most violated constraint.

    As for finding the most violated constraint, it is enough to compute
    joint_feature(x, y_hat), not djoint_feature, we can optionally skip
    computing joint_feature(x, y) using compute_differences=False
    """
    h = model.latent(x, y, w)
    h_hat = model.loss_augmented_inference(x, h, w, relaxed=relaxed)
    joint_feature = model.joint_feature
    delta_joint_feature = joint_feature(x, h) - joint_feature(x, h_hat)

    loss = model.loss(y, h_hat)
    slack = max(loss - np.dot(w, delta_joint_feature), 0)
    return h_hat, delta_joint_feature, slack, loss


def inference(model, x, w):
    return model.inference(x, w)


def loss_augmented_inference(model, x, y, w, relaxed=True):
    return model.loss_augmented_inference(x, y, w, relaxed=relaxed)


# easy debugging
def objective_primal(model, w, X, Y, C, variant='n_slack', n_jobs=1):
    objective = 0
    constraints = Parallel(
        n_jobs=n_jobs)(delayed(find_constraint)(
            model, x, y, w)
            for x, y in zip(X, Y))
    slacks = list(zip(*constraints))[2]

    if variant == 'n_slack':
        slacks = np.maximum(slacks, 0)

    objective = max(np.sum(slacks), 0) * C + np.sum(w ** 2) / 2.
    return objective


def exhaustive_loss_augmented_inference(model, x, y, w):
    size = y.size
    best_y = None
    best_energy = np.inf
    for y_hat in itertools.product(range(model.n_states), repeat=size):
        y_hat = np.array(y_hat).reshape(y.shape)
        #print("trying %s" % repr(y_hat))
        joint_feature = model.joint_feature(x, y_hat)
        energy = -model.loss(y, y_hat) - np.dot(w, joint_feature)
        if energy < best_energy:
            best_energy = energy
            best_y = y_hat
    return best_y


def exhaustive_inference(model, x, w):
    # hack to get the grid shape of x
    if isinstance(x, np.ndarray):
        feats = x
    else:
        feats = model._get_features(x)
    size = np.prod(feats.shape[:-1])
    best_y = None
    best_energy = np.inf
    for y_hat in itertools.product(range(model.n_states), repeat=size):
        y_hat = np.array(y_hat).reshape(feats.shape[:-1])
        #print("trying %s" % repr(y_hat))
        joint_feature = model.joint_feature(x, y_hat)
        energy = -np.dot(w, joint_feature)
        if energy < best_energy:
            best_energy = energy
            best_y = y_hat
    return best_y
