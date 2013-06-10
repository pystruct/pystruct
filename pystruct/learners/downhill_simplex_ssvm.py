import numpy as np
from scipy.optimize import fmin

from .cutting_plane_ssvm import NSlackSSVM
from ..utils import find_constraint


class PrimalDSNSlackSSVM(NSlackSSVM):
    """Uses downhill simplex for optimizing an unconstraint primal.

    This is basically a sanity check on all other implementations,
    as this is easier to check for correctness.
    """

    def fit(self, X, Y):
        def func(w):
            objective = 0
            for x, y in zip(X, Y):
                y_hat, delta_psi, slack, loss = find_constraint(self.model,
                                                                x, y, w)
                objective += slack
            objective /= float(len(X))
            objective += np.sum(w ** 2) / float(self.C) / 2.
            return objective
        w = 1e-5 * np.ones(self.model.size_psi)
        res = fmin(func, x0=w + 1, full_output=1)
        res2 = fmin(func, x0=w, full_output=1)
        self.w = res[0] if res[1] < res2[1] else res2[0]
        return self
