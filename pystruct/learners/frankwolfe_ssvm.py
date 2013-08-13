from pystruct.learners.ssvm import BaseSSVM
import numpy as np
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
        lam = 1.0 / self.C
        ls = 0
        ws = 0.0
        n_samples = len(X)
        n_pos_slack = 0
        for x, y in zip(X, Y):
            y_hat, delta_psi, slack, loss = find_constraint(self.model, x, y, self.w)

            ws += delta_psi
            ls += loss
            if slack > 0:
                n_pos_slack += 1
        ws /= (lam * n_samples)
        ls /= n_samples

        dual_val = -0.5 * lam * np.sum(self.w ** 2) + l
        w_diff = self.w - ws
        dual_gap = lam * w_diff.T.dot(self.w) - l + ls
        primal_val = dual_val + dual_gap
        return dual_val, dual_gap, primal_val, n_pos_slack

    def _frank_wolfe_batch(self, X, Y):
        # Algorithm 2: Batch Frank-Wolfe
        l = 0.0
        n_samples = float(len(X))
        lam = 1.0 / self.C
        for k in xrange(self.max_iter):
            ls = 0
            ws = np.zeros(self.model.size_psi)
            n_pos_slack = 0
            for x, y in zip(X, Y):
                y_hat, delta_psi, slack, loss = find_constraint(self.model, x, y, self.w)
                ws += (delta_psi / (lam * n_samples))
                ls += (loss / n_samples)
                if slack > 0:
                    n_pos_slack += 1

            w_diff = self.w - ws
            dual_gap = lam * w_diff.T.dot(self.w) - l + ls

            # line search for gamma
            if self.line_search:
                eps = 2.2204e-16
                gamma = dual_gap / (lam * (np.sum(w_diff ** 2)) + eps)
                gamma = max(0.0, min(1.0, gamma))
            else:
                gamma = 2.0 / (k + 2.0)

            # update w and l
            self.w = (1.0 - gamma) * self.w + gamma * ws
            l = (1.0 - gamma) * l + gamma * ls

            dual_val, dual_gap, primal_val, n_pos_slack = self._calc_dual_gap(X, Y, l)
            print("k = %d, dual: %f, dual_gap: %f, primal: %f, gamma: %f, n_pos_slack: %f"
                  % (k, dual_val, dual_gap, primal_val, gamma, n_pos_slack))
            if dual_gap < self.tol:
                return

    def _frank_wolfe_bc(self, X, Y):
        # Algorithm 3: block-coordinate Frank-Wolfe
        n_samples = len(X)
        w_mat = np.zeros((n_samples, self.model.size_psi))
        l_mat = np.zeros(n_samples)

        lam = 1.0 / self.C
        l = 0
        k = 0
        for p in xrange(self.max_iter):
            if self.verbose > 0:
                print("Iteration %d" % p)
            for i in range(n_samples):
                x, y = X[i], Y[i]
                y_hat, delta_psi, slack, loss = find_constraint(self.model, x, y, self.w)
                # ws and ls
                ws = delta_psi / (n_samples * lam)
                ls = loss / n_samples

                # line search
                if self.line_search:
                    eps = 1e-15
                    w_diff = w_mat[i] - ws
                    gamma = (w_diff.T.dot(self.w) - 1.0/lam*(l_mat[i] - ls)) / (np.sum(w_diff ** 2) + eps)
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

                if (self.dual_check_every != 0) and (k % self.dual_check_every == 0):
                    dual_val, dual_gap, primal_val, n_pos_slack = self._calc_dual_gap(X, Y, l)
                    print("p = %d, dual: %f, dual_gap: %f, primal: %f, positive slack: %d"
                          % (p, dual_val, dual_gap, primal_val, n_pos_slack))
                    if dual_gap < self.tol:
                        return

    def fit(self, X, Y):
        self.model.initialize(X, Y)
        self.w = getattr(self, "w", np.zeros(self.model.size_psi))
        if self.batch_mode:
            self._frank_wolfe_batch(X, Y)
        else:
            self._frank_wolfe_bc(X, Y)
        return self
