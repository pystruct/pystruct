######################
# Authors:
#   Xianghang Liu <xianghangliu@gmail.com>
#   Andreas Mueller <amueller@ais.uni-bonn.de>
#
# License: BSD 3-clause
#
# Implements structured SVM as described in Joachims et. al.
# Cutting-Plane Training of Structural SVMs

from time import time
import numpy as np

from pystruct.learners.ssvm import BaseSSVM
from pystruct.utils import find_constraint


class FrankWolfeSSVM(BaseSSVM):
    """Structured SVM solver using Block-coordinate Frank-Wolfe.

    This implementation is somewhat experimental. Use with care.

    This implementation follows the paper:
        Lacoste-Julien, Jaggi, Schmidt, Pletscher JMLR 2013
        Block-Coordinage Frank-Wolfe Optimization for Structural SVMs

    With batch_mode=False, this implements the online (block-coordinate)
    version of the algorithm (BCFW)
    BCFW is an attractive alternative to subgradient methods, as no
    learning rate is needed and a duality gap guarantee is given.

    Parameters
    ----------
    model : StructuredModel
        Object containing the model structure. Has to implement
        `loss`, `inference` and `loss_augmented_inference`.

    max_iter : int, default=1000
        Maximum number of passes over dataset to find constraints.

    C : float, default=1
        Regularization parameter. Corresponds to 1 / (lambda * n_samples).

    verbose : int
        Verbosity.

    n_jobs : int, default=1
        Number of parallel processes. Currently only n_jobs=1 is supported.

    show_loss_every : int, default=0
        How often the training set loss should be computed.
        Zero corresponds to never.

    tol : float, default=1e-3
        Convergence tolerance on the duality gap.

    logger : logger object, default=None
        Pystruct logger for storing the model or extracting additional
        information.

    batch_mode : boolean, default=False
        Whether to use batch updates. Will slow down learning enormously.

    line_search : boolean, default=True
        Whether to compute the optimum step size in each step.
        The line-search is done in closed form and cheap.
        There is usually no reason to turn this off.

    check_dual_every : int, default=10
        How often the stopping criterion should be checked. Computing
        the stopping criterion is as costly as doing one pass over the dataset,
        so check_dual_every=1 will make learning twice as slow.

    do_averaging : bool, default=True
        Whether to use weight averaging as described in the reference paper.
        Currently this is only supported in the block-coordinate version.


    Attributes
    ----------
    w : nd-array, shape=(model.size_psi,)
        The learned weights of the SVM.

    ``loss_curve_`` : list of float
        List of loss values if show_loss_every > 0.

    ``objective_curve_`` : list of float
       Cutting plane objective after each pass through the dataset.

    ``primal_objective_curve_`` : list of float
        Primal objective after each pass through the dataset.

    ``timestamps_`` : list of int
       Total training time stored before each iteration.
    """
    def __init__(self, model, max_iter=1000, C=1.0, verbose=0, n_jobs=1,
                 show_loss_every=0, logger=None, batch_mode=False,
                 line_search=True, check_dual_every=10, tol=.001,
                 do_averaging=True):

        if n_jobs != 1:
            raise ValueError("FrankWolfeSSVM does not support multiprocessing"
                             " yet. Ignoring n_jobs != 1.")
        BaseSSVM.__init__(self, model, max_iter, C, verbose=verbose,
                          n_jobs=n_jobs, show_loss_every=show_loss_every,
                          logger=logger)
        self.tol = tol
        self.batch_mode = batch_mode
        self.line_search = line_search
        self.check_dual_every = check_dual_every
        self.do_averaging = do_averaging

    def _calc_dual_gap(self, X, Y, l):
        n_samples = len(X)
        psi_gt = self.model.batch_psi(X, Y, Y)  # FIXME don't calculate this again
        Y_hat = self.model.batch_loss_augmented_inference(X, Y, self.w,
                                                          relaxed=True)
        dpsi = psi_gt - self.model.batch_psi(X, Y_hat)
        ls = np.sum(self.model.batch_loss(Y, Y_hat))
        ws = dpsi * self.C
        l = l * n_samples * self.C

        dual_val = -0.5 * np.sum(self.w ** 2) + l
        w_diff = self.w - ws
        dual_gap = w_diff.T.dot(self.w) - l + ls * self.C
        primal_val = dual_val + dual_gap
        self.primal_objective_curve_.append(primal_val)
        self.objective_curve_.append(dual_val)
        self.timestamps_.append(time() - self.timestamps_[0])
        return dual_val, dual_gap, primal_val

    def _frank_wolfe_batch(self, X, Y):
        """Batch Frank-Wolfe learning.

        This is basically included for reference / comparision only,
        as the block-coordinate version is much faster.

        Compare Algorithm 2 in the reference paper.
        """
        l = 0.0
        n_samples = float(len(X))
        psi_gt = self.model.batch_psi(X, Y, Y)

        for k in xrange(self.max_iter):
            ls = 0
            Y_hat = self.model.batch_loss_augmented_inference(X, Y, self.w,
                                                              relaxed=True)
            dpsi = psi_gt - self.model.batch_psi(X, Y_hat)
            ls = np.mean(self.model.batch_loss(Y, Y_hat))
            ws = dpsi * self.C

            w_diff = self.w - ws
            dual_gap = 1.0 / (self.C * n_samples)* w_diff.T.dot(self.w) - l + ls

            # line search for gamma
            if self.line_search:
                eps = 1e-15
                gamma = dual_gap / (np.sum(w_diff ** 2) / (self.C * n_samples) + eps)
                gamma = max(0.0, min(1.0, gamma))
            else:
                gamma = 2.0 / (k + 2.0)

            # update w and l
            self.w = (1.0 - gamma) * self.w + gamma * ws
            l = (1.0 - gamma) * l + gamma * ls

            if (self.check_dual_every != 0) and (k % self.check_dual_every == 0):
                # FIXME we shouldn't need to recompute everything here, right?
                dual_val, dual_gap, primal_val = self._calc_dual_gap(X, Y, l)
                if self.verbose > 0:
                    print("k = %d, dual: %f, dual_gap: %f, primal: %f, gamma: %f"
                          % (k, dual_val, dual_gap, primal_val, gamma))
                if dual_gap < self.tol:
                    return

    def _frank_wolfe_bc(self, X, Y):
        """Block-Coordinate Frank-Wolfe learning.

        Compare Algorithm 3 in the reference paper.
        """
        n_samples = len(X)
        w = self.w.copy()
        w_mat = np.zeros((n_samples, self.model.size_psi))
        l_mat = np.zeros(n_samples)
        l_avg = 0.0
        l = 0.0
        k = 0
        for p in xrange(self.max_iter):
            if self.verbose > 0:
                print("Iteration %d" % p)
            for i in range(n_samples):
                x, y = X[i], Y[i]
                y_hat, delta_psi, slack, loss = find_constraint(self.model, x, y, w)
                # ws and ls
                ws = delta_psi * self.C
                ls = loss / n_samples

                # line search
                if self.line_search:
                    eps = 1e-15
                    w_diff = w_mat[i] - ws
                    gamma = (w_diff.T.dot(w) - (self.C * n_samples)*(l_mat[i] - ls)) / (np.sum(w_diff ** 2) + eps)
                    gamma = max(0.0, min(1.0, gamma))
                else:
                    gamma = 2.0 * n_samples / (k + 2.0 * n_samples)

                w -= w_mat[i]
                w_mat[i] = (1.0 - gamma) * w_mat[i] + gamma * ws
                w += w_mat[i]

                l -= l_mat[i]
                l_mat[i] = (1.0 - gamma) * l_mat[i] + gamma * ls
                l += l_mat[i]

                if self.do_averaging:
                    rho = 2.0 / (k + 2.0)
                    self.w = (1.0 - rho) * self.w + rho * w
                    l_avg = (1.0 - rho) * l_avg + rho * l
                else:
                    self.w = w
                k += 1

            if (self.check_dual_every != 0) and (p % self.check_dual_every == 0):
                dual_val, dual_gap, primal_val = self._calc_dual_gap(X, Y, l)
                if self.verbose > 0:
                    print("dual: %f, dual_gap: %f, primal: %f"
                          % (dual_val, dual_gap, primal_val))
                if dual_gap < self.tol:
                    return

    def fit(self, X, Y, constraints=None, initialize=True):
        """Learn parameters using (block-coordinate) Frank-Wolfe learning.

        Parameters
        ----------
        X : iterable
            Traing instances. Contains the structured input objects.
            No requirement on the particular form of entries of X is made.

        Y : iterable
            Training labels. Contains the strctured labels for inputs in X.
            Needs to have the same length as X.

        contraints : ignored

        initialize : boolean, default=True
            Whether to initialize the model for the data.
            Leave this true except if you really know what you are doing.
        """
        if initialize:
            self.model.initialize(X, Y)
        self.objective_curve_, self.primal_objective_curve_ = [], []
        self.timestamps_ = [time()]
        self.w = getattr(self, "w", np.zeros(self.model.size_psi))
        try:
            if self.batch_mode:
                self._frank_wolfe_batch(X, Y)
            else:
                self._frank_wolfe_bc(X, Y)
        except KeyboardInterrupt:
            pass
        return self
