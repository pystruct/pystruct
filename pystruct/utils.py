import itertools
import cPickle

import numpy as np


def unwrap_pairwise(y):
    """given a y that may contain pairwise marginals, yield plain y."""
    if isinstance(y, tuple):
        return y[0]
    return y


def make_grid_edges(x, neighborhood=4, return_lists=False):
    if neighborhood not in [4, 8]:
        raise ValueError("neighborhood can only be '4' or '8', got %s" %
                         repr(neighborhood))
    inds = np.arange(x.shape[0] * x.shape[1]).reshape(x.shape[:2])
    inds = inds.astype(np.int64)
    right = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
    down = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
    edges = [right, down]
    if neighborhood == 8:
        upright = np.c_[inds[1:, :-1].ravel(), inds[:-1, 1:].ravel()]
        downright = np.c_[inds[:-1, :-1].ravel(), inds[1:, 1:].ravel()]
        edges.extend([upright, downright])
    if return_lists:
        return edges
    return np.vstack(edges)


def compute_energy(x, y, unary_params, pairwise_params, neighborhood=4):
    # x is unaries
    # y is a labeling
    n_states = x.shape[-1]
    if isinstance(y, tuple):
        # y can also be continuous (from lp)
        # in this case, it comes with accumulated edge marginals
        y, pw = y
        x_flat = x.reshape(-1, x.shape[-1])
        y_flat = y.reshape(-1, y.shape[-1])
        unaries_acc = np.sum(x_flat * y_flat, axis=0)
        pw = pw.reshape(-1, n_states, n_states).sum(axis=0)
    else:
        ## unary features:
        gx, gy = np.ogrid[:x.shape[0], :x.shape[1]]
        selected_unaries = x[gx, gy, y]
        unaries_acc = np.bincount(y.ravel(), selected_unaries.ravel(),
                                  minlength=n_states)

        ##accumulated pairwise
        #make one hot encoding
        labels = np.zeros((y.shape[0], y.shape[1], n_states),
                          dtype=np.int)
        labels[gx, gy, y] = 1

        if neighborhood == 4:
            # vertical edges
            vert = np.dot(labels[1:, :, :].reshape(-1, n_states).T,
                          labels[:-1, :, :].reshape(-1, n_states))
            # horizontal edges
            horz = np.dot(labels[:, 1:, :].reshape(-1, n_states).T,
                          labels[:, :-1, :].reshape(-1, n_states))
            pw = vert + horz
        elif neighborhood == 8:
            # vertical edges
            vert = np.dot(labels[1:, :, :].reshape(-1, n_states).T,
                          labels[:-1, :, :].reshape(-1, n_states))
            # horizontal edges
            horz = np.dot(labels[:, 1:, :].reshape(-1, n_states).T,
                          labels[:, :-1, :].reshape(-1, n_states))
            diag1 = np.dot(labels[1:, 1:, :].reshape(-1, n_states).T,
                           labels[1:, :-1, :].reshape(-1, n_states))
            diag2 = np.dot(labels[1:, 1:, :].reshape(-1, n_states).T,
                           labels[:-1, :-1, :].reshape(-1, n_states))
            pw = vert + horz + diag1 + diag2
    pw = pw + pw.T - np.diag(np.diag(pw))
    energy = (np.dot(unaries_acc, unary_params)
              + np.dot(np.tril(pw).ravel(), pairwise_params.ravel()))
    return energy


## global functions for easy parallelization
def find_constraint(model, x, y, w, y_hat=None, relaxed=True,
                    compute_difference=True):
    """Find most violated constraint, or, given y_hat,
    find slack and dpsi for this constraing.

    As for finding the most violated constraint, it is enough to compute
    psi(x, y_hat), not dpsi, we can optionally skip computing psi(x, y)
    using compute_differences=False
    """

    if y_hat is None:
        y_hat = model.loss_augmented_inference(x, y, w, relaxed=relaxed)
    psi = model.psi
    if compute_difference:
        delta_psi = psi(x, y) - psi(x, y_hat)
    else:
        delta_psi = -psi(x, y_hat)
    if isinstance(y_hat, tuple):
        # continuous label
        loss = model.continuous_loss(y, y_hat[0])
    else:
        loss = model.loss(y, y_hat)
    slack = max(loss - np.dot(w, delta_psi), 0)
    return y_hat, delta_psi, slack, loss


def find_constraint_latent(model, x, y, w, relaxed=True):
    """Find most violated constraint.

    As for finding the most violated constraint, it is enough to compute
    psi(x, y_hat), not dpsi, we can optionally skip computing psi(x, y)
    using compute_differences=False
    """
    h = model.latent(x, y, w)
    h_hat = model.loss_augmented_inference(x, h, w, relaxed=relaxed)
    psi = model.psi
    delta_psi = psi(x, h) - psi(x, h_hat)

    loss = model.loss(y, h_hat)
    slack = max(loss - np.dot(w, delta_psi), 0)
    return h_hat, delta_psi, slack, loss


def inference(model, x, w):
    return model.inference(x, w)


def loss_augmented_inference(model, x, y, w, relaxed=True):
    return model.loss_augmented_inference(x, y, w, relaxed=relaxed)


# easy debugging
def objective_primal(model, w, X, Y, C):
    objective = 0
    psi = model.psi
    for x, y in zip(X, Y):
        y_hat = model.loss_augmented_inference(x, y, w)
        loss = model.loss(y, y_hat)
        delta_psi = psi(x, y) - psi(x, y_hat)
        objective += loss - np.dot(w, delta_psi)
    objective /= float(len(X))
    objective += np.sum(w ** 2) / float(C) / 2.
    return objective


def exhaustive_loss_augmented_inference(model, x, y, w):
    size = y.size
    best_y = None
    best_energy = np.inf
    for y_hat in itertools.product(range(model.n_states), repeat=size):
        y_hat = np.array(y_hat).reshape(y.shape)
        #print("trying %s" % repr(y_hat))
        psi = model.psi(x, y_hat)
        energy = -model.loss(y, y_hat) - np.dot(w, psi)
        if energy < best_energy:
            best_energy = energy
            best_y = y_hat
    return best_y


def exhaustive_inference(model, x, w):
    # hack to get the grid shape of x
    if isinstance(x, np.ndarray):
        feats = x
    else:
        feats = model.get_features(x)
    size = np.prod(feats.shape[:-1])
    best_y = None
    best_energy = np.inf
    for y_hat in itertools.product(range(model.n_states), repeat=size):
        y_hat = np.array(y_hat).reshape(feats.shape[:-1])
        #print("trying %s" % repr(y_hat))
        psi = model.psi(x, y_hat)
        energy = -np.dot(w, psi)
        if energy < best_energy:
            best_energy = energy
            best_y = y_hat
    return best_y


class SaveLogger(object):
    def __init__(self, file_name, save_every=10, verbose=0):
        self.file_name = file_name
        self.save_every = save_every
        self.verbose = verbose

    def __repr__(self):
        return ('%s(file_name="%s", save_every=%s)'
                % (self.__class__.__name__, self.file_name, self.save_every))

    def __call__(self, learner, iteration=0):
        if iteration == 'final' or not iteration % self.save_every:
            file_name = self.file_name
            if "%d" in file_name:
                file_name = file_name % iteration
            if self.verbose > 0:
                print("saving %s to file %s" % (learner, file_name))
            with open(file_name, "wb") as f:
                if hasattr(learner, 'inference_cache_'):
                    # don't store the large inference cache!
                    learner.inference_cache_, tmp = (None,
                                                     learner.inference_cache_)
                    cPickle.dump(learner, f, -1)
                    learner.inference_cache_ = tmp
                else:
                    cPickle.dump(learner, f, -1)

    def load(self):
        with open(self.file_name, "rb") as f:
            learner = cPickle.load(f)
        return learner
