import numpy as np
import cvxopt
import cvxopt.solvers
from IPython.core.debugger import Tracer

tracer = Tracer()


class StructuredSVM(object):
    def __init__(self, problem, max_iter=100, C=1.0):
        self.max_iter = max_iter
        self.problem = problem
        self.C = float(C)

    def fit(self, X, Y):
        psi = self.problem.psi
        w = np.zeros(self.problem.size_psi)
        constraints = []
        losses = []
        psis = []
        for iteration in xrange(self.max_iter):
            print("iteration %d" % iteration)
            new_constraints = 0
            for x, y in zip(X, Y):
                y_hat = self.problem.inference(x, w)
                if (y_hat != y).any():
                    constraints.append((x, y, y_hat))
                    delta_psi = psi(x, y) - psi(x, y_hat)
                    psis.append(delta_psi / 1000.)
                    losses.append(self.problem.loss(y, y_hat))
                    new_constraints += 1
                else:
                    print("found gt!")
            print("mean loss: %f" % np.mean(losses))
            if new_constraints == 0:
                break

            psi_matrix = np.vstack(psis)
            n_constraints = len(constraints)
            P = cvxopt.matrix(np.dot(psi_matrix, psi_matrix.T))
            q = cvxopt.matrix(-np.ones(n_constraints))
            idy = np.identity(n_constraints)
            G = cvxopt.matrix(np.vstack((-idy, idy)))
            tmp1 = np.zeros(n_constraints)
            tmp2 = np.ones(n_constraints) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
            # solve QP problem
            solution = cvxopt.solvers.qp(P, q, G, h)

            # Lagrange multipliers
            a = np.ravel(solution['x'])

            # Support vectors have non zero lagrange multipliers
            sv = a > 1e-5
            print("%d support vectors out of %d points" % (np.sum(sv),
                n_constraints))
            w = np.zeros(self.problem.size_psi)
            for issv, dpsi, alpha in zip(sv, psis, a):
                if not issv:
                    continue
                w += alpha * dpsi
            print(w)
        self.w = w

    def predict(self, X):
        prediction = []
        for x in X:
            prediction.append(self.problem.inference(x, self.w))
        return prediction
