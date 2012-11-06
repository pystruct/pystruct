######################
# (c) 2012 Andreas Mueller <amueller@ais.uni-bonn.de>
# License: BSD 3-clause
#
# Implements structured SVM as described in Tsochantaridis et. al.
# Support Vector Machines Learning for Interdependent
# and Structures Output Spaces

import numpy as np
#from numpy.testing import assert_almost_equal
import cvxopt
import cvxopt.solvers
import matplotlib.pyplot as plt
from scipy.optimize import fmin

from IPython.core.debugger import Tracer
tracer = Tracer()


def objective_primal(problem, w, X, Y, C):
    objective = 0
    psi = problem.psi
    for x, y in zip(X, Y):
        y_hat = problem.loss_augmented_inference(x, y, w)
        loss = problem.loss(y, y_hat)
        delta_psi = psi(x, y) - psi(x, y_hat)
        objective += loss - np.dot(w, delta_psi)
    objective /= float(len(X))
    objective += np.sum(w ** 2) / float(C) / 2.
    return objective


class StructuredSVM(object):
    """Structured SVM training with l1 slack penalty.

    Implements slack and margin rescaled structural SVM using
    the dual formulation and cutting plane method, solved using CVXOPT.
    The optimization is restarted in each iteration, therefore
    possibly leading to a large overhead.

    Parameters
    ----------
    problem : StructuredProblem
        Object containing problem formulation. Has to implement
        `loss`, `inference` and `loss_augmented_inference`.
    max_iter : int
        Maximum number of passes over dataset to find constraints.
    C : float
        Regularization parameter
    check_constraints : bool
        Whether to check if the new "most violated constraint" is
        more violated than previous constraints. Helpful for stopping
        and debugging, but costly.
    verbose : int
        Verbosity
    positive_constraint: list of ints
        Indices of parmeters that are constraint to be positive.
    """

    def __init__(self, problem, max_iter=100, C=1.0, check_constraints=True,
            verbose=1, positive_constraint=None):
        self.max_iter = max_iter
        self.positive_constraint = positive_constraint
        self.problem = problem
        self.C = float(C)
        self.verbose = verbose
        self.check_constraints = check_constraints
        if verbose == 0:
            cvxopt.solvers.options['show_progress'] = False

    def _solve_constraints(self, constraints, n_samples):
        C = self.C / float(n_samples)
        psis = [c[1] for sample in constraints for c in sample]
        losses = [c[2] for sample in constraints for c in sample]
        psi_matrix = np.vstack(psis)
        n_constraints = len(psis)
        P = cvxopt.matrix(np.dot(psi_matrix, psi_matrix.T))
        q = cvxopt.matrix(-np.array(losses, dtype=np.float))
        # constraints are a bit tricky. first, all learningrate must be >zero
        idy = np.identity(n_constraints)
        tmp1 = np.zeros(n_constraints)
        # box constraint: sum of all learningrate for one example must be <= C
        blocks = np.zeros((n_samples, n_constraints))
        first = 0
        for i, sample in enumerate(constraints):
            blocks[i, first: first + len(sample)] = 1
            first += len(sample)
        # positivity constraints:
        if self.positive_constraint is None:
            #empty constraints
            zero_constr = np.zeros(0)
            psis_constr = np.zeros((0, n_constraints))
        else:
            psis_constr = psi_matrix.T[self.positive_constraint]
            zero_constr = np.zeros(len(self.positive_constraint))

        # put together
        G = cvxopt.matrix(np.vstack((-idy, blocks, psis_constr)))
        tmp2 = np.ones(n_samples) * C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2, zero_constr)))
        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h)

        # Lagrange multipliers
        a = np.ravel(solution['x'])
        self.old_solution = solution

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        box = np.dot(blocks, a)
        if self.verbose > 1:
            print("%d support vectors out of %d points" % (np.sum(sv),
                n_constraints))
            # calculate per example box constraint:
            print("Box constraints at C: %d" % np.sum(1 - box / C < 1e-3))
            print("dual objective: %f" % solution['primal objective'])
        w = np.dot(a, psi_matrix)
        return w, solution['primal objective']

    def _find_constraint(self, x, y, w, y_hat=None):
        """Find most violated constraint, or, given y_hat,
        find slack and dpsi for this constraing."""

        if y_hat is None:
            y_hat = self.problem.loss_augmented_inference(x, y, w)
        psi = self.problem.psi
        loss = self.problem.loss(y, y_hat)
        delta_psi = psi(x, y) - psi(x, y_hat)
        slack = loss - np.dot(w, delta_psi)
        return y_hat, delta_psi, slack, loss

    def fit(self, X, Y):
        print("Training dual structural SVM")
        # we initialize with a small value so that loss-augmented inference
        # can give us something meaningful in the first iteration
        w = np.ones(self.problem.size_psi) * 1e-5
        n_samples = len(X)
        constraints = [[] for i in xrange(n_samples)]
        loss_curve = []
        objective_curve = []
        primal_objective_curve = []
        for iteration in xrange(self.max_iter):
            if self.verbose > 0:
                print("iteration %d" % iteration)
            new_constraints = 0
            current_loss = 0.
            primal_objective = 0.
            for i, x, y in zip(np.arange(len(X)), X, Y):
                y_hat, delta_psi, slack, loss = self._find_constraint(x, y, w)

                if self.verbose > 1:
                    print("current slack: %f" % slack)
                primal_objective += slack

                already_active = np.any([True for y_hat_, psi_, loss_ in
                    constraints[i] if (y_hat == y_hat_).all()])
                if already_active:
                    continue

                if self.check_constraints:
                    # "smart" but expensive stopping criterion
                    # check if most violated constraint is more violated
                    # than previous ones by more then eps.
                    # If it is less violated, inference was wrong/approximate
                    for con in constraints[i]:
                        # compute slack for old constraint
                        slack_tmp = con[2] - np.dot(w, con[1])
                        if self.verbose > 1:
                            print("slack old constraint: %f" % slack_tmp)
                        # if slack of new constraint is smaller or not
                        # significantly larger, don't add constraint.
                        # if smaller, complain about approximate inference.
                        if (slack - slack_tmp) < -1e-5:
                            print("bad inference!")
                            #tracer()
                            already_active = True
                            break

                current_loss += loss
                # if significant slack and constraint not active
                # this is a weaker check than the "check_constraints" one.
                if not already_active and slack > 1e-5:
                    constraints[i].append([y_hat, delta_psi, loss])
                    new_constraints += 1
            primal_objective /= len(X)
            current_loss /= len(X)
            primal_objective += np.sum(w ** 2) / self.C / 2.
            #assert_almost_equal(primal_objective,
                    #objective_primal(self.problem, w, X, Y, self.C))
            if self.verbose > 0:
                print("current loss: %f  new constraints: %d, primal obj: %f" %
                        (current_loss, new_constraints, primal_objective))
            loss_curve.append(current_loss)

            primal_objective_curve.append(primal_objective)
            if new_constraints == 0:
                print("no additional constraints")
                break
            w, objective = self._solve_constraints(constraints, n_samples)
            objective_curve.append(objective)
            #if (iteration > 1 and np.abs(objective_curve[-2] -
                #objective_curve[-1]) < 0.01):
                #print("Dual objective converged.")
                #break
            if self.verbose > 0:
                print(w)
        self.w = w
        self.constraints_ = constraints
        print("calls to inference: %d" % self.problem.inference_calls)
        #plt.figure()
        #plt.subplot(131, title="loss")
        #plt.plot(loss_curve)
        #plt.subplot(132, title="objective")
        # the objective value should be monotonically decreasing
        # this is a maximization problem, to which we add more
        # and more constraints
        #plt.plot(objective_curve)
        #plt.subplot(133, title="primal objective")
        #plt.plot(primal_objective_curve)
        #plt.show()
        #plt.close()
        self.primal_objective_ = primal_objective

    def predict(self, X):
        prediction = []
        for x in X:
            prediction.append(self.problem.inference(x, self.w))
        return prediction


class PrimalDSStructuredSVM(StructuredSVM):
    """Uses downhill simplex for optimizing an unconstraint primal.

    This is basically a sanity check on all other implementations,
    as this is easier to check for correctness.
    """

    def fit(self, X, Y):
        def func(w):
            objective = 0
            for x, y in zip(X, Y):
                y_hat, delta_psi, slack, loss = self._find_constraint(x, y, w)
                objective += slack
            objective /= float(len(X))
            objective += np.sum(w ** 2) / float(self.C) / 2.
            return objective
        w = 1e-5 * np.ones(self.problem.size_psi)
        res = fmin(func, x0=w + 1, full_output=1)
        res2 = fmin(func, x0=w, full_output=1)
        self.w = res[0] if res[1] < res2[1] else res2[0]
        tracer()
        return self


class SubgradientStructuredSVM(StructuredSVM):
    """Margin rescaled with l1 slack penalty."""
    def __init__(self, problem, max_iter=100, C=1.0, verbose=0, momentum=0.9,
            learningrate=0.001, plot=False):
        super(SubgradientStructuredSVM, self).__init__(problem, max_iter, C,
                verbose=verbose)
        self.momentum = momentum
        self.learningrate = learningrate
        self.t = 0
        self.plot = plot

    def _solve_subgradient(self, psis):
        if hasattr(self, 'w'):
            w = self.w
        else:
            w = np.zeros(self.problem.size_psi)
            self.grad_old = np.zeros(self.problem.size_psi)
        psi_matrix = np.vstack(psis).mean(axis=0)
        #w += 1. / self.t * (psi_matrix - w / self.C / 2)
        #grad = (self.learningrate / (self.t + 1.) ** 2
                #* (psi_matrix - w / self.C / 2))
        grad = self.learningrate * (psi_matrix - w / self.C / 2)
        w += grad + self.momentum * self.grad_old
        self.grad_old = grad
        self.w = w
        self.t += 1.
        return w

    def fit(self, X, Y):
        print("Training primal subgradient structural SVM")
        # we initialize with a small value so that loss-augmented inference
        # can give us something meaningful in the first iteration
        w = 1e-5 * np.ones(self.problem.size_psi)
        #constraints = []
        all_psis = []
        losses = []
        loss_curve = []
        objective_curve = []
        for iteration in xrange(self.max_iter):
            psis = []
            positive_slacks = 0
            current_loss = 0.
            objective = 0.
            for i, x, y in zip(np.arange(len(X)), X, Y):
                y_hat, delta_psi, slack, loss = self._find_constraint(x, y, w)
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
            if self.verbose > 0:
                print("iteration %d" % iteration)
                print("current loss: %f  positive slacks: %d, objective: %f" %
                        (current_loss, positive_slacks, objective))
            loss_curve.append(current_loss)
            all_psis.extend(psis)
            objective_curve.append(objective)
            w = self._solve_subgradient(psis)

            if self.verbose > 2:
                print(w)
        self.w = w
        print("final objective: %f" % objective_curve[-1])
        print("calls to inference: %d" % self.problem.inference_calls)
        if self.plot:
            plt.subplot(121, title="loss")
            plt.plot(loss_curve)
            plt.subplot(122, title="objective")
            plt.plot(objective_curve)
            plt.show()
