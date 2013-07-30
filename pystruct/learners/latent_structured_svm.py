######################
# (c) 2012 Andreas Mueller <amueller@ais.uni-bonn.de>
#
#


import numpy as np

from .ssvm import BaseSSVM
from .n_slack_ssvm import NSlackSSVM
from ..utils import find_constraint


class LatentSSVM(BaseSSVM):
    """Stuctured SVM solver for latent-variable models.

    Alternates between doing latent variale completion and solving the
    completed SSVM problem using the ``base_ssvm`` solver.

    The model is expected to know how to initialize itself
    - it should provide a ``init_latent`` procedure.

    If the base_ssvm is an n-slack SSVM, the current constraints
    will be adjusted after recomputing the latent variables H.
    If the base_ssvm is a 1-slack SSVM, the inference cache will
    be reused. Both methods drastically speed up learning.

    Parameters
    ----------
    base_ssvm : object
        SSVM solver instance for solving the completed problem.

    latent_iter : int (default=5)
        Number of iterations in the completion / refit loop.

    logger : object
        Logger instance.

    Attributes
    ----------
    w : nd-array, shape=(model.size_psi,)
        The learned weights of the SVM.
    """

    def __init__(self, base_ssvm, latent_iter=5, logger=None):
        self.base_ssvm = base_ssvm
        self.latent_iter = latent_iter
        self.logger = logger

    def fit(self, X, Y, H_init=None):
        self.model.initialize(X, Y)
        w = np.zeros(self.model.size_psi)
        constraints = None
        ws = []
        if H_init is None:
            H_init = self.model.init_latent(X, Y)
        self.H_init_ = H_init
        H = H_init

        for iteration in xrange(self.latent_iter):
            print("LATENT SVM ITERATION %d" % iteration)
            # find latent variables for ground truth:
            if iteration == 0:
                pass
            else:
                H_new = [self.model.latent(x, y, w) for x, y in zip(X, Y)]
                changes = [np.any(h_new != h) for h_new, h in zip(H_new, H)]
                if not np.any(changes):
                    print("no changes in latent variables of ground truth."
                          " stopping.")
                    break
                print("changes in H: %d" % np.sum(changes))

                # update constraints:
                if isinstance(self.base_ssvm, NSlackSSVM):
                    constraints = [[] for i in xrange(len(X))]
                    for sample, h, i in zip(self.base_ssvm.constraints_, H_new,
                                            np.arange(len(X))):
                        for constraint in sample:
                            const = find_constraint(self.model, X[i], h, w,
                                                    constraint[0])
                            y_hat, dpsi, _, loss = const
                            constraints[i].append([y_hat, dpsi, loss])
                H = H_new
            if iteration > 0:
                self.base_ssvm.fit(X, H, constraints=constraints,
                                   warm_start="soft", initialize=False)
            else:
                self.base_ssvm.fit(X, H, constraints=constraints,
                                   initialize=False)
            w = self.base_ssvm.w
            ws.append(w)
            if self.logger is not None:
                self.logger(self, iteration)

    def predict(self, X):
        prediction = self.base_ssvm.predict(X)
        return [self.model.label_from_latent(h) for h in prediction]

    def predict_latent(self, X):
        return self.base_ssvm.predict(X)

    def score(self, X, Y):
        """Compute score as 1 - loss over whole data set.

        Returns the average accuracy (in terms of model.loss)
        over X and Y.

        Parameters
        ----------
        X : iterable
            Evaluation data.

        Y : iterable
            True labels.

        Returns
        -------
        score : float
            Average of 1 - loss over training examples.
        """
        losses = [self.model.base_loss(y, y_pred)
                  for y, y_pred in zip(Y, self.predict(X))]
        max_losses = [self.model.max_loss(y) for y in Y]
        return 1. - np.sum(losses) / float(np.sum(max_losses))

    @property
    def model(self):
        return self.base_ssvm.model

    @model.setter
    def model(self, model_):
        self.base_ssvm.model = model_

    @property
    def w(self):
        return self.base_ssvm.w

    @w.setter
    def w(self, w_):
        self.base_ssvm.w = w_

    @property
    def C(self):
        return self.base_ssvm.C

    @C.setter
    def C(self, C_):
        self.base_ssvm.w = C_

    @property
    def n_jobs(self):
        return self.base_ssvm.n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs_):
        self.base_ssvm.n_jobs = n_jobs_
