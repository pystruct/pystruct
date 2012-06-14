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

tracer = Tracer()


class StructuredSVM(object):
    """Margin rescaled with l1 slack penalty."""
    def __init__(self, problem, max_iter=100, C=1.0, check_constraints=True,
            verbose=1):
        self.max_iter = max_iter
        self.problem = problem
        self.C = float(C)
        self.verbose = verbose
        self.check_constraints = check_constraints
        if verbose == 0:
            cvxopt.solvers.options['show_progress'] = False

    def _solve_qp(self, constraints, n_samples):
        C = self.C / float(n_samples)
        psis = [c[1] for sample in constraints for c in sample]
        losses = [c[2] for sample in constraints for c in sample]
        psi_matrix = np.vstack(psis)
        n_constraints = len(psis)
        P = cvxopt.matrix(np.dot(psi_matrix, psi_matrix.T))
        q = cvxopt.matrix(-np.array(losses, dtype=np.float))
        # constraints are a bit tricky. first, all alpha must be >zero
        idy = np.identity(n_constraints)
        tmp1 = np.zeros(n_constraints)
        # box constraint: sum of all alpha for one example must be <= C
        blocks = np.zeros((n_samples, n_constraints))
        first = 0
        for i, sample in enumerate(constraints):
            blocks[i, first: first + len(sample)] = 1
            first += len(sample)
        tracer()
        # put together
        G = cvxopt.matrix(np.vstack((-idy, blocks)))
        tmp2 = np.ones(n_samples) * C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h)

        # Lagrange multipliers
        a = np.ravel(solution['x'])
        self.old_solution = solution

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        print("%d support vectors out of %d points" % (np.sum(sv),
            n_constraints))
        print("Coefficients at C: %d" % np.sum(1 - a / C < 1e-3))
        print("dual objective: %f" % solution['primal objective'])
        w = np.zeros(self.problem.size_psi)
        for issv, dpsi, alpha in zip(sv, psis, a):
            if not issv:
                continue
            w += alpha * dpsi
        return w, solution['primal objective']

    def fit(self, X, Y):
        psi = self.problem.psi
        w = np.zeros(self.problem.size_psi)
        n_samples = len(X)
        constraints = [[] for i in xrange(n_samples)]
        loss_curve = []
        objective_curve = []
        primal_objective_curve = []
        for iteration in xrange(self.max_iter):
            print("iteration %d" % iteration)
            new_constraints = 0
            current_loss = 0.
            primal_objective = 0.
            for i, x, y in zip(np.arange(len(X)), X, Y):
                y_hat = self.problem.loss_augmented_inference(x, y, w)
                loss = self.problem.loss(y, y_hat)

                already_active = np.any([True for y_hat_, psi_, loss_ in
                    constraints[i] if (y_hat == y_hat_).all()])

                delta_psi = psi(x, y) - psi(x, y_hat)
                slack = loss - np.dot(w, delta_psi)
                primal_objective += slack
                if self.verbose > 1:
                    print("current slack: %f" % slack)
                if self.check_constraints:
                    # "smart" but expensive stopping criterion
                    # check if most violated constraint is more violated
                    # than previous ones by more then eps.
                    # If it is less violated, inference was wrong/approximate
                    for con in constraints[i]:
                        # compute slack for old constraint
                        dpsi_tmp = psi(x, y) - psi(x, con[0])
                        loss_tmp = self.problem.loss(y, con[0])
                        slack_tmp = loss_tmp - np.dot(w, dpsi_tmp)
                        if self.verbose > 1:
                            print("slack old constraint: %f" % slack_tmp)
                        if slack < slack_tmp:
                            print("bad inference!")
                            already_active = True
                        if (slack - slack_tmp) < 1e-5:
                            already_active = True

                current_loss += loss
                #if loss and not already_active:
                if not already_active and slack > 1e-5:
                    constraints[i].append([y_hat, delta_psi, loss])
                    new_constraints += 1
            primal_objective /= len(X)
            current_loss /= len(X)
            primal_objective += np.sum(w ** 2) / self.C / 2.
            print("current loss: %f  new constraints: %d, primal obj: %f" %
                    (current_loss, new_constraints, primal_objective))
            loss_curve.append(current_loss)

            primal_objective_curve.append(primal_objective)
            if new_constraints == 0:
                print("no additional constraints")
                break
            w, objective = self._solve_qp(constraints, n_samples)
            objective_curve.append(objective)
            if (iteration > 1 and np.abs(objective_curve[-2] -
                objective_curve[-1]) < 0.01):
                print("Dual objective converged.")
                break
            if self.verbose > 0:
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
        plt.plot(primal_objective_curve)
        plt.show()
        tracer()

    def predict(self, X):
        prediction = []
        for x in X:
            prediction.append(self.problem.inference(x, self.w))
        return prediction


class SubgradientStructuredSVM(StructuredSVM):
    """Margin rescaled with l1 slack penalty."""
    def __init__(self, problem, max_iter=100, C=1.0):
        super(SubgradientStructuredSVM, self).__init__(problem, max_iter, C)
        self.t = 1000.

    def _solve_subgradient(self, psis):
        if hasattr(self, 'w'):
            w = self.w
        else:
            w = np.zeros(self.problem.size_psi)
        psi_matrix = np.vstack(psis).mean(axis=0)
        w += 1. / self.t * (psi_matrix - w / self.C / 2)
        #w += .01 * (psi_matrix - w / self.C / 2)
        self.w = w
        self.t += 1.
        return w

    def fit(self, X, Y):
        psi = self.problem.psi
        w = np.zeros(self.problem.size_psi)
        #constraints = []
        all_psis = []
        losses = []
        loss_curve = []
        objective_curve = []
        for iteration in xrange(self.max_iter):
            print("iteration %d" % iteration)
            psis = []
            positive_slacks = 0
            current_loss = 0.
            objective = 0.
            for i, x, y in zip(np.arange(len(X)), X, Y):
                y_hat = self.problem.loss_augmented_inference(x, y, w)
                loss = self.problem.loss(y, y_hat)
                delta_psi = psi(x, y) - psi(x, y_hat)
                slack = loss - np.dot(delta_psi, w)
                objective += slack
                psis.append(delta_psi)

                losses.append(loss)
                current_loss += loss
                if slack > 0:
                    positive_slacks += 1
            objective /= len(X)
            current_loss /= len(X)
            objective += np.sum(w ** 2) / self.C / 2.
            if positive_slacks == 0:
                print("No additional constraints")
                break
            print("current loss: %f  positive slacks: %d, objective: %f" %
                    (current_loss, positive_slacks, objective))
            loss_curve.append(current_loss)
            all_psis.extend(psis)
            objective_curve.append(objective)
            w = self._solve_subgradient(psis)
            if self.verbose > 2:
                print(w)
        self.w = w
        print(objective_curve[-1])
        plt.subplot(121, title="loss")
        plt.plot(loss_curve[10:])
        plt.subplot(122, title="objective")
        plt.plot(objective_curve[10:])
        plt.show()


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
            current_loss /= len(X)
            print("current loss: %f  new constraints: %d" %
                    (current_loss, new_constraints))
            loss_curve.append(current_loss)
            w = self._solve_qp(psis, losses)

            print(w)
        self.w = w
