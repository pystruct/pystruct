######################
# (c) 2012 Andreas Mueller <amueller@ais.uni-bonn.de>
# License: BSD 3-clause
#
# Implements structured SVM as described in Joachims et. al.
# Cutting-Plane Training of Structural SVMs

import numpy as np
import cvxopt
import cvxopt.solvers

from sklearn.externals.joblib import Parallel, delayed

from .ssvm import BaseSSVM
from ..utils import loss_augmented_inference


class OneSlackSSVM(BaseSSVM):
    """Structured SVM training with l1 slack penalty.

    Implements margin rescaled structural SVM using
    the 1-slack formulation and cutting plane method, solved using CVXOPT.
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

    break_on_bad: bool (default=True)
        Whether to break (start debug mode) when inference was approximate.

    n_jobs : int, default=1
        Number of parallel jobs for inference. -1 means as many as cpus.

    show_loss : string, default='augmented'
        Controlls the meaning of the loss curve and convergence messages.
        By default (show_loss='augmented') the loss of the loss-augmented
        prediction is shown, since this is computed any way.
        Setting show_loss='real' will show the true loss, i.e. the one of
        the normal prediction. Be aware that this means an additional
        call to inference in each iteration!

    batch_size : int, default=100
        Number of constraints after which we solve the QP again.

    tol : float, default=-10
        Convergence tolerance. If dual objective decreases less than tol,
        learning is stopped. The default corresponds to ignoring the behavior
        of the dual objective and stop only if no more constraints can be
        found.


    Attributes
    ----------
    w : nd-array, shape=(problem.psi,)
        The learned weights of the SVM.

    old_solution : dict
        The last solution found by the qp solver.

   ``loss_curve_`` : list of float
        List of loss values after each pass thorugh the dataset.
        Either sum of slacks (loss on loss augmented predictions)
        or actual loss, depends on the value of ``show_loss``.

   ``objective_curve_`` : list of float
       Primal objective after each pass through the dataset.
    """

    def __init__(self, problem, max_iter=100, C=1.0, check_constraints=True,
                 verbose=1, positive_constraint=None, n_jobs=1,
                 break_on_bad=True, show_loss='true', tol=0.0001):

        BaseSSVM.__init__(self, problem, max_iter, C, verbose=verbose,
                          n_jobs=n_jobs, show_loss=show_loss)

        self.positive_constraint = positive_constraint
        self.check_constraints = check_constraints
        self.break_on_bad = break_on_bad
        self.tol = tol

    def _solve_1_slack_qp(self, constraints, n_samples):
        C = np.float(self.C)
        psis = [c[0] for c in constraints]
        losses = [c[1] for c in constraints]

        psi_matrix = np.vstack(psis)
        n_constraints = len(psis)
        P = cvxopt.matrix(np.dot(psi_matrix, psi_matrix.T))
        # q contains loss from margin-rescaling
        q = cvxopt.matrix(-np.array(losses, dtype=np.float))
        # constraints: all alpha must be >zero
        idy = np.identity(n_constraints)
        tmp1 = np.zeros(n_constraints)
        # positivity constraints:
        if self.positive_constraint is None:
            #empty constraints
            zero_constr = np.zeros(0)
            psis_constr = np.zeros((0, n_constraints))
        else:
            psis_constr = psi_matrix.T[self.positive_constraint]
            zero_constr = np.zeros(len(self.positive_constraint))

        # put together
        G = cvxopt.sparse(cvxopt.matrix(np.vstack((-idy, psis_constr))))
        h = cvxopt.matrix(np.hstack((tmp1, zero_constr)))

        # equality constraint: sum of all alpha must be = C
        A = cvxopt.matrix(np.ones((1, n_constraints)))
        b = cvxopt.matrix([C])

        # solve QP problem
        cvxopt.solvers.options['feastol'] = 1e-5
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        if solution['status'] != "optimal":
            print("regularizing QP!")
            P = cvxopt.matrix(np.dot(psi_matrix, psi_matrix.T)
                              + 1e-8 * np.eye(psi_matrix.shape[0]))
            solution = cvxopt.solvers.qp(P, q, G, h, A, b)
            if solution['status'] != "optimal":
                from IPython.core.debugger import Tracer
                Tracer()()

        # Lagrange multipliers
        a = np.ravel(solution['x'])
        self.alphas.append(a)
        self.old_solution = solution

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-10
        if self.verbose > 1:
            print("%d support vectors out of %d points" % (np.sum(sv),
                                                           n_constraints))
        w = np.dot(a, psi_matrix)
        return w, solution['primal objective']

    def _check_bad_constraint(self, slack, dpsi_mean, loss,
                              old_constraints, w):
        if slack < 1e-5:
            return True
        #Ys_plain = [unwrap_pairwise(y) for y in Ys]
        #all_old_Ys = [[unwrap_pairwise(y_) for y_ in Ys_]
                      #for Ys_, _, _ in old_constraints]
        #equals = [np.all([np.all(y == y_) for y, y_ in zip(Ys_plain, old_Ys)])
                  #for old_Ys in all_old_Ys]
        equals = [True for dpsi_, loss_ in old_constraints
                  if (np.all(dpsi_ == dpsi_mean) and loss == loss_)]

        if np.any(equals):
            return True

        if self.check_constraints:
            for con in old_constraints:
                # compute slack for old constraint
                slack_tmp = max(con[1] - np.dot(w, con[0]), 0)
                if self.verbose > 5:
                    print("slack old constraint: %f" % slack_tmp)
                # if slack of new constraint is smaller or not
                # significantly larger, don't add constraint.
                # if smaller, complain about approximate inference.
                if slack - slack_tmp < -1e-5:
                    print("bad inference: %f" % (slack_tmp - slack))
                    if self.break_on_bad:
                        from IPython.core.debugger import Tracer
                        Tracer()()
                    return True
        return False

    def fit(self, X, Y, constraints=None):
        """Learn parameters using cutting plane method.

        Parameters
        ----------
        X : iterable
            Traing instances. Contains the structured input objects.
            No requirement on the particular form of entries of X is made.

        Y : iterable
            Training labels. Contains the strctured labels for inputs in X.
            Needs to have the same length as X.

        contraints : iterable
            Known constraints for warm-starts. List of same length as X.
            Each entry is itself a list of constraints for a given instance x .
            Each constraint is of the form [y_hat, delta_psi, loss], where
            y_hat is a labeling, ``delta_psi = psi(x, y) - psi(x, y_hat)``
            and loss is the loss for predicting y_hat instead of the true label
            y.
        """
        print("Training 1-slack dual structural SVM")
        if self.verbose < 2:
            cvxopt.solvers.options['show_progress'] = False
        else:
            cvxopt.solvers.options['show_progress'] = True
        w = np.zeros(self.problem.size_psi)
        n_samples = len(X)
        if constraints is None:
            constraints = []
        loss_curve = []
        objective_curve = []
        self.alphas = []  # dual solutions
        # we have to update at least once after going through the dataset
        for iteration in xrange(self.max_iter):
            # main loop
            if self.verbose > 0:
                print("iteration %d" % iteration)
            # do inference in parallel
            verbose = max(0, self.verbose - 3)
            if self.n_jobs != 1:
                verbose = max(0, self.verbose - 3)
                Y_hat = Parallel(n_jobs=self.n_jobs, verbose=verbose)(
                    delayed(loss_augmented_inference)(
                        self.problem, x, y, w) for x, y in zip(X, Y))
            else:
                Y_hat = [self.problem.loss_augmented_inference(x, y, w)
                         for x, y in zip(X, Y)]

            # compute the mean over psis and losses
            dpsi_mean = np.zeros(self.problem.size_psi)
            for x, y, y_hat in zip(X, Y, Y_hat):
                dpsi_mean += self.problem.psi(x, y)
                dpsi_mean -= self.problem.psi(x, y_hat)
            dpsi_mean /= n_samples

            loss_mean = np.mean([self.problem.loss(y, y_hat)
                                 for y, y_hat in zip(Y, Y_hat)])

            slack = loss_mean - np.dot(w, dpsi_mean)

            if self.verbose > 0:
                if self.show_loss == 'true':
                    display_loss = np.mean([
                        self.problem.loss(y, self.problem.inference(x, w))
                        for y, x in zip(Y, X)])
                else:
                    display_loss = loss_mean
                print("current loss: %f  new slack: %f"
                      % (display_loss, slack))
            # now check the slack + the constraint
            if self._check_bad_constraint(slack, dpsi_mean, loss_mean,
                                          constraints, w):
                print("no additional constraints")
                break

            constraints.append((dpsi_mean, loss_mean))

            w, objective = self._solve_1_slack_qp(constraints,
                                                  n_samples)
            if self.verbose > 0:
                print("dual objective: %f" % objective)
            objective_curve.append(objective)

            loss_curve.append(loss_mean)

            if (iteration > 1 and objective_curve[-2]
                    - objective_curve[-1] < self.tol):
                print("objective converged.")
                break
            if self.verbose > 5:
                print(w)
        self.w = w
        self.constraints_ = constraints
        print("calls to inference: %d" % self.problem.inference_calls)
        self.loss_curve_ = loss_curve
        self.objective_curve_ = objective_curve
