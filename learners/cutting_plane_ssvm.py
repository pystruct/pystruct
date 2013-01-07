######################
# (c) 2012 Andreas Mueller <amueller@ais.uni-bonn.de>
# License: BSD 3-clause
#
# Implements structured SVM as described in Tsochantaridis et. al.
# Support Vector Machines Learning for Interdependent
# and Structures Output Spaces

import numpy as np
import cvxopt
import cvxopt.solvers
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

from ..utils import unwrap_pairwise, find_constraint

from IPython.core.debugger import Tracer
tracer = Tracer()


class StructuredSVM(object):
    """Structured SVM training with l1 slack penalty.

    Implements margin rescaled structural SVM using
    the n-slack formulation and cutting plane method, solved using CVXOPT.
    The optimization is restarted in each iteration.

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

    plot : bool (default=Fale)
        Whether to plot a learning curve in the end.

    break_on_bad: bool (default=True)
        Whether to break (start debug mode) when inference was approximate.
    """

    def __init__(self, problem, max_iter=100, C=1.0, check_constraints=True,
                 verbose=1, positive_constraint=None, n_jobs=1, plot=False,
                 break_on_bad=True):
        self.max_iter = max_iter
        self.positive_constraint = positive_constraint
        self.problem = problem
        self.C = float(C)
        self.verbose = verbose
        self.check_constraints = check_constraints
        self.n_jobs = n_jobs
        self.plot = plot
        self.break_on_bad = break_on_bad
        if verbose < 2:
            cvxopt.solvers.options['show_progress'] = False

    def _solve_n_slack_qp(self, constraints, n_samples):
        C = self.C / float(n_samples)
        psis = [c[1] for sample in constraints for c in sample]
        losses = [c[2] for sample in constraints for c in sample]
        psi_matrix = np.vstack(psis)
        n_constraints = len(psis)
        P = cvxopt.matrix(np.dot(psi_matrix, psi_matrix.T))
        # q contains loss from margin-rescaling
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
        cvxopt.solvers.options['feastol'] = 1e-5
        solution = cvxopt.solvers.qp(P, q, G, h)
        if solution['status'] != "optimal":
            print("regularizing QP!")
            P = cvxopt.matrix(np.dot(psi_matrix, psi_matrix.T)
                              + 1e-8 * np.eye(psi_matrix.shape[0]))
            solution = cvxopt.solvers.qp(P, q, G, h)
            if solution['status'] != "optimal":
                tracer()

        # Lagrange multipliers
        a = np.ravel(solution['x'])
        self.alphas.append(a)
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

    def fit(self, X, Y, constraints=None):
        print("Training dual structural SVM")
        # we initialize with a small value so that loss-augmented inference
        # can give us something meaningful in the first iteration
        w = np.ones(self.problem.size_psi) * 1e-5
        n_samples = len(X)
        if constraints is None:
            constraints = [[] for i in xrange(n_samples)]
        loss_curve = []
        objective_curve = []
        primal_objective_curve = []
        self.ws = []
        self.alphas = []  # dual solutions
        for iteration in xrange(self.max_iter):
            if self.verbose > 0:
                print("iteration %d" % iteration)
            new_constraints = 0
            current_loss = 0.
            #for i, x, y in zip(np.arange(len(X)), X, Y):
                #y_hat, delta_psi, slack, loss = self._find_constraint(x, y, w)
            candidate_constraints = (Parallel(n_jobs=self.n_jobs)
                                     (delayed(find_constraint)
                                      (self.problem, x, y, w)
                                      for x, y in zip(X, Y)))
            for i, x, y, constraint in zip(np.arange(len(X)), X, Y,
                                           candidate_constraints):
                y_hat, delta_psi, slack, loss = constraint

                current_loss += loss

                if self.verbose > 1:
                    print("current slack: %f" % slack)
                y_hat_plain = unwrap_pairwise(y_hat)
                already_active = np.any([True for y__, _, _ in constraints[i]
                                         if (y_hat_plain ==
                                             unwrap_pairwise(y__)).all()])
                if already_active:
                    continue

                if self.check_constraints:
                    # "smart" stopping criterion
                    # check if most violated constraint is more violated
                    # than previous ones by more then eps.
                    # If it is less violated, inference was wrong/approximate
                    for con in constraints[i]:
                        # compute slack for old constraint
                        slack_tmp = max(con[2] - np.dot(w, con[1]), 0)
                        if self.verbose > 1:
                            print("slack old constraint: %f" % slack_tmp)
                        # if slack of new constraint is smaller or not
                        # significantly larger, don't add constraint.
                        # if smaller, complain about approximate inference.
                        if slack - slack_tmp < -1e-5:
                            print("bad inference: %f" % (slack_tmp - slack))
                            if self.break_on_bad:
                                tracer()
                            already_active = True
                            break

                # if significant slack and constraint not active
                # this is a weaker check than the "check_constraints" one.
                if not already_active and slack > 1e-5:
                    constraints[i].append([y_hat, delta_psi, loss])
                    new_constraints += 1
            current_loss /= len(X)
            loss_curve.append(current_loss)

            if new_constraints == 0:
                print("no additional constraints")
                #tracer()
                if iteration > 0:
                    break
            w, objective = self._solve_n_slack_qp(constraints, n_samples)

            # hack to make loss-augmented prediction working:
            unary_weights = self.problem.get_unary_weights(w)
            unary_weights[unary_weights == 0] = 1e-10
            #w[:self.problem.n_states][w[:self.problem.n_states] == 0] = 1e-10
            slacks = [max(np.max([-np.dot(w, psi_) + loss_
                                  for _, psi_, loss_ in sample]), 0)
                      for sample in constraints]
            sum_of_slacks = np.sum(slacks)
            objective_p = self.C * sum_of_slacks / len(X) + np.sum(w ** 2) / 2.
            primal_objective_curve.append(objective_p)
            if (len(primal_objective_curve) > 2
                    and objective_p > primal_objective_curve[-2] + 1e8):
                print("primal loss became smaller. that shouldn't happen.")
                tracer()
            objective_curve.append(objective)
            if self.verbose > 0:
                print("current loss: %f  new constraints: %d, "
                      "primal objective: %f dual objective: %f" %
                      (current_loss, new_constraints,
                       primal_objective_curve[-1], objective))
            if (iteration > 1 and primal_objective_curve[-1] -
                    primal_objective_curve[-2] < 0.0001):
                print("objective converged.")
                break
            self.ws.append(w)
            if self.verbose > 1:
                print(w)
        self.w = w
        self.constraints_ = constraints
        print("calls to inference: %d" % self.problem.inference_calls)
        if self.plot:
            plt.figure()
            plt.subplot(131, title="loss")
            plt.plot(loss_curve)
            plt.subplot(132, title="objective")
            plt.plot(objective_curve)
            plt.subplot(133, title="primal objective")
            plt.plot(primal_objective_curve)
            plt.show()
            plt.close()
        #self.primal_objective_ = primal_objective_curve[-1]

    def predict(self, X):
        prediction = []
        for x in X:
            prediction.append(self.problem.inference(x, self.w))
        return prediction
