from time import time
import numpy as np

from pystruct.learners.ssvm import BaseSSVM
from pystruct.utils import find_constraint


class FrankWolfeSSVM(BaseSSVM):
    def __init__(self, model, max_iter=1000, C=1.0, verbose=0, n_jobs=1,
                 show_loss_every=0, logger=None, batch_mode=True,
                 line_search=True, dual_check_every=1, tol=.001):

        BaseSSVM.__init__(self, model, max_iter, C, verbose=verbose,
                          n_jobs=n_jobs, show_loss_every=show_loss_every,
                          logger=logger)
        self.tol = tol
        self.batch_mode = batch_mode
        self.line_search = line_search
        self.dual_check_every = dual_check_every

    def _calc_dual_gap(self, X, Y, l):
        n_samples = len(X)
        ls = 0
        ws = 0.0
        n_pos_slack = 0
        for x, y in zip(X, Y):
            y_hat, delta_psi, slack, loss = find_constraint(self.model, x, y, self.w)

            ws += delta_psi
            ls += loss
            if slack > 0:
                n_pos_slack += 1
        ws *= self.C
        l = l * n_samples * self.C

        dual_val = -0.5  * np.sum(self.w ** 2) + l
        w_diff = self.w - ws
        dual_gap = w_diff.T.dot(self.w) - l + ls * self.C
        primal_val = dual_val + dual_gap
        self.primal_objective_curve_.append(primal_val)
        self.objective_curve_.append(dual_val)
        self.timestamps_.append(time() - self.timestamps_[0])
        return dual_val, dual_gap, primal_val, n_pos_slack

    def _frank_wolfe_batch(self, X, Y):
        # Algorithm 2: Batch Frank-Wolfe
        l = 0.0
        n_samples = float(len(X))
        for k in xrange(self.max_iter):
            ls = 0
            ws = np.zeros(self.model.size_psi)
            n_pos_slack = 0
            for x, y in zip(X, Y):
                y_hat, delta_psi, slack, loss = find_constraint(self.model, x, y, self.w)
                ws += delta_psi  * self.C
                ls += (loss / n_samples)
                if slack > 0:
                    n_pos_slack += 1

            w_diff = self.w - ws
            dual_gap = 1.0 / (self.C * n_samples)* w_diff.T.dot(self.w) - l + ls

            # line search for gamma
            if self.line_search:
                eps = 2.2204e-16
                gamma = dual_gap / (np.sum(w_diff ** 2) / (self.C * n_samples) + eps)
                gamma = max(0.0, min(1.0, gamma))
            else:
                gamma = 2.0 / (k + 2.0)

            # update w and l
            self.w = (1.0 - gamma) * self.w + gamma * ws
            l = (1.0 - gamma) * l + gamma * ls

            dual_val, dual_gap, primal_val, n_pos_slack = self._calc_dual_gap(X, Y, l)
            if self.verbose > 0:
                print("k = %d, dual: %f, dual_gap: %f, primal: %f, gamma: %f, n_pos_slack: %f"
                      % (k, dual_val, dual_gap, primal_val, gamma, n_pos_slack))
            if dual_gap < self.tol:
                return

    def _frank_wolfe_bc(self, X, Y):
        # Algorithm 3: block-coordinate Frank-Wolfe
        n_samples = len(X)
        w_mat = np.zeros((n_samples, self.model.size_psi))
        l_mat = np.zeros(n_samples)

        l = 0
        k = 0
        for p in xrange(self.max_iter):
            if self.verbose > 0:
                print("Iteration %d" % p)
            for i in range(n_samples):
                x, y = X[i], Y[i]
                y_hat, delta_psi, slack, loss = find_constraint(self.model, x, y, self.w)
                # ws and ls
                ws = delta_psi * self.C
                ls = loss / n_samples

                # line search
                if self.line_search:
                    eps = 1e-15
                    w_diff = w_mat[i] - ws
                    gamma = (w_diff.T.dot(self.w) - (self.C * n_samples)*(l_mat[i] - ls)) / (np.sum(w_diff ** 2) + eps)
                    gamma = max(0.0, min(1.0, gamma))
                else:
                    gamma = 2.0 * n_samples / (k + 2.0 * n_samples)

                self.w -= w_mat[i]
                w_mat[i] = (1.0 - gamma) * w_mat[i] + gamma * ws
                self.w += w_mat[i]

                l -= l_mat[i]
                l_mat[i] = (1.0 - gamma) * l_mat[i] + gamma * ls
                l += l_mat[i]

                k += 1
                if self.verbose > 1:
                    print("Approximate loss: %f" %l)

        if (self.dual_check_every != 0) and (p % self.dual_check_every == 0):
            dual_val, dual_gap, primal_val, n_pos_slack = self._calc_dual_gap(X, Y, l)
            if self.verbose > 0:
                print("dual: %f, dual_gap: %f, primal: %f, positive slack: %d"
                      % (dual_val, dual_gap, primal_val, n_pos_slack))
            if dual_gap < self.tol:
                return

    def fit(self, X, Y, constraints=None, initialize=True):
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
