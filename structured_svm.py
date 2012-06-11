######################
# (c) 2012 Andreas Mueller <amueller@ais.uni-bonn.de>
# License: BSD 3-clause
#
# Implements structured SVM as described in Tsochantaridis et. al.
# Support Vector Machines Learning for Interdependend
# and Structures Output Spaces

import numpy as np
import cvxopt
import cvxopt.solvers
import matplotlib.pyplot as plt

from IPython.core.debugger import Tracer

#cvxopt.solvers.options['show_progress'] = False
tracer = Tracer()


class StructuredSVM(object):
    """Margin rescaled with l1 slack penalty."""
    def __init__(self, problem, max_iter=100, C=1.0):
        self.max_iter = max_iter
        self.problem = problem
        self.C = float(C)

    def _solve_qp(self, psis, losses):
        psi_matrix = np.vstack(psis)
        n_constraints = len(psis)
        P = cvxopt.matrix(np.dot(psi_matrix, psi_matrix.T))
        q = cvxopt.matrix(-np.array(losses, dtype=np.float))
        idy = np.identity(n_constraints)
        G = cvxopt.matrix(np.vstack((-idy, idy)))
        tmp1 = np.zeros(n_constraints)
        tmp2 = np.ones(n_constraints) * self.C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
        # get previous solution as starting point if any
        initvals = dict()
        if hasattr(self, "old_solution"):
            initvals['x'] = self.old_solution['x']
            initvals['y'] = self.old_solution['y']
        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h)

        # Lagrange multipliers
        a = np.ravel(solution['x'])
        self.old_solution = solution

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        print("%d support vectors out of %d points" % (np.sum(sv),
            n_constraints))
        print("Coefficients at C: %d" % np.sum(1 - a / self.C < 1e-3))
        print("primal objective: %f" % solution['primal objective'])
        w = np.zeros(self.problem.size_psi)
        for issv, dpsi, alpha in zip(sv, psis, a):
            if not issv:
                continue
            w += alpha * dpsi
        return w, solution['primal objective']

    def fit(self, X, Y):
        psi = self.problem.psi
        w = np.zeros(self.problem.size_psi)
        constraints = []
        psis = []
        losses = []
        loss_curve = []
        objective_curve = []
        real_objective_curve = []
        for iteration in xrange(self.max_iter):
            print("iteration %d" % iteration)
            new_constraints = 0
            current_loss = 0.
            real_objective = np.sum(w ** 2)
            for i, x, y in zip(np.arange(len(X)), X, Y):
                y_hat = self.problem.loss_augmented_inference(x, y, w)
                loss = self.problem.loss(y, y_hat)

                already_active = [True for i_, y_hat_ in constraints if
                        i_ == i and (y_hat == y_hat_).all()]
                constraint = (i, y_hat)
                delta_psi = psi(x, y) - psi(x, y_hat)
                real_objective -= np.dot(w, delta_psi) - loss
                if loss and not already_active:
                    constraints.append(constraint)
                    psis.append(delta_psi)
                    losses.append(loss)
                    current_loss += loss
                    new_constraints += 1
            if new_constraints == 0:
                print("no additional constraints")
                break
            print("current loss: %f  new constraints: %d, real obj: %f" %
                    (current_loss / len(X), new_constraints, real_objective))
            loss_curve.append(current_loss / len(X))
            w, objective = self._solve_qp(psis, losses)
            objective_curve.append(objective)
            real_objective_curve.append(real_objective)
            #if iteration > 1 and np.abs(objective_curve[-2] - objective_curve[-1]) < 0.01:
                #print("Dual objective converged.")
                #break

            print(w)
        self.w = w
        plt.subplot(131, title="loss")
        plt.plot(loss_curve)
        plt.subplot(132, title="objective")
        # the objective value should be monotonically decreasing
        # this is a maximization problem, to which we add more
        # and more constraints
        plt.plot(objective_curve)
        plt.subplot(133, title="primal objective")
        plt.plot(real_objective_curve)
        plt.show()

    def predict(self, X):
        prediction = []
        for x in X:
            prediction.append(self.problem.inference(x, self.w))
        return prediction


class LatentStructuredSVM(StructuredSVM):
    """Margin rescaled with l1 slack penalty."""
    def fit(self, X, Y):
        psi = self.problem.psi
        w = np.zeros(self.problem.size_psi)
        constraints = []
        losses = []
        loss_curve = []
        for iteration in xrange(self.max_iter):
            print("iteration %d" % iteration)
            psis = []
            # recompute psi from previous constraints
            for i, h_hat, y_hat in constraints:
                x, y = X[i], Y[i]
                h = self.problem.latent(x, y, w)
                #h_hat = self.problem.latent(x, y_hat, w)
                delta_psi = psi(x, h, y) - psi(x, h_hat, y_hat)
                psis.append(delta_psi)
            new_constraints = 0
            current_loss = 0.
            for i, x, y in zip(np.arange(len(X)), X, Y):
                h = self.problem.latent(x, y, w)
                h_hat, y_hat = self.problem.loss_augmented_inference(x, y, w)
                if i < 5 and not iteration % 10:
                    plt.matshow(h.reshape(18, 18))
                    plt.colorbar()
                    plt.savefig("figures/h_%03d_%03d.png" % (iteration, i))
                    plt.close()
                    plt.matshow(h_hat.reshape(18, 18))
                    plt.colorbar()
                    plt.savefig("figures/h_hat_%03d_%03d.png" % (iteration, i))
                    plt.close()
                loss = self.problem.loss(y, y_hat)
                constraint = (i, h_hat, y_hat)
                already_active = [True for i_, h_hat_, y_hat_ in constraints if
                        i_ == i and (y_hat == y_hat_).all()
                        and (h_hat == h_hat_).all()]
                if already_active:
                    print("ASDF")

                if loss and not already_active:
                    current_loss += loss

                    if not already_active:
                        constraints.append(constraint)
                        delta_psi = psi(x, h, y) - psi(x, h_hat, y_hat)
                        psis.append(delta_psi)
                        losses.append(loss)
                        new_constraints += 1

            if new_constraints == 0:
                print("no additional constraints found")
                tracer()
                break
            print("current loss: %f  new constraints: %d" %
                    (current_loss / len(X), new_constraints))
            loss_curve.append(current_loss)
            w = self._solve_qp(psis, losses)

            print(w)
        self.w = w

    def predict(self, X):
        prediction = []
        for x in X:
            prediction.append(self.problem.inference(x, self.w)[0])
        return prediction
