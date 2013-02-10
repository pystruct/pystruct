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


class NoConstraint(Exception):
    # raised if we can not construct a constraint from cache
    pass


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

    show_loss_every : int, default=0
        Controlls how often the hamming loss is computed (for monitoring
        purposes). Zero means never, otherwise it will be computed very
        show_loss_every'th epoch.

    batch_size : int, default=100
        Number of constraints after which we solve the QP again.

    tol : float, default=-10
        Convergence tolerance. If dual objective decreases less than tol,
        learning is stopped. The default corresponds to ignoring the behavior
        of the dual objective and stop only if no more constraints can be
        found.

    inference_cache : int, default=0
        How many results of loss_augmented_inference to cache per sample.  If >
        0 the most violating of the cached examples will be used to construct a
        global constraint. Only if this constraint is not violated, inference
        will be run again.


    Attributes
    ----------
    w : nd-array, shape=(problem.psi,)
        The learned weights of the SVM.

    old_solution : dict
        The last solution found by the qp solver.

   ``loss_curve_`` : list of float
        List of loss values if show_loss_every > 0.

   ``objective_curve_`` : list of float
       Primal objective after each pass through the dataset.
    """

    def __init__(self, problem, max_iter=100, C=1.0, check_constraints=True,
                 verbose=1, positive_constraint=None, n_jobs=1,
                 break_on_bad=True, show_loss_every=0, tol=0.0001,
                 inference_cache=0):

        BaseSSVM.__init__(self, problem, max_iter, C, verbose=verbose,
                          n_jobs=n_jobs, show_loss_every=show_loss_every)

        self.positive_constraint = positive_constraint
        self.check_constraints = check_constraints
        self.break_on_bad = break_on_bad
        self.tol = tol
        self.inference_cache = inference_cache

    def _solve_1_slack_qp(self, constraints, n_samples):
        C = np.float(self.C) * n_samples  # this is how libsvm/svmstruct do it
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
        sv = a > 1e-5
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

    def _update_cache(self, Y_hat):
        """Updated cached constraints."""
        if not self.inference_cache:
            return
        for sample, y_hat in zip(self.inference_cache_, Y_hat):
            if len(sample) > self.inference_cache:
                sample.pop(0)
            sample.append(y_hat)

    def _constraint_from_cache(self, X, Y, w, psi_gt, constraints):
        if not self.inference_cache:
            raise NoConstraint
        Y_hat = []
        for x, y, cached in zip(X, Y, self.inference_cache_):
            violations = [np.dot(self.problem.psi(x, y_hat), w)
                          + self.problem.loss(y, y_hat)
                          for y_hat in cached]
            Y_hat.append(cached[np.argmax(violations)])

        dpsi = (psi_gt - self.problem.batch_psi(X, Y_hat)) / len(X)
        loss_mean = np.mean(self.problem.batch_loss(Y, Y_hat))

        slack = loss_mean - np.dot(w, dpsi)
        if self._check_bad_constraint(slack, dpsi, loss_mean,
                                      constraints, w):
            raise NoConstraint
        if self.verbose > 0:
            print("new slack: %f" % (slack))
        return Y_hat

    def _find_new_constraint(self, X, Y, w, psi_gt, constraints):
        if self.n_jobs != 1:
            # do inference in parallel
            verbose = max(0, self.verbose - 3)
            Y_hat = Parallel(n_jobs=self.n_jobs, verbose=verbose)(
                delayed(loss_augmented_inference)(
                    self.problem, x, y, w, relaxed=True)
                for x, y in zip(X, Y))
        else:
            Y_hat = self.problem.batch_loss_augmented_inference(
                X, Y, w, relaxed=True)
        # compute the mean over psis and losses

        dpsi = (psi_gt - self.problem.batch_psi(X, Y_hat)) / len(X)
        loss_mean = np.mean(self.problem.batch_loss(Y, Y_hat))

        slack = loss_mean - np.dot(w, dpsi)
        if self._check_bad_constraint(slack, dpsi, loss_mean,
                                      constraints, w):
            raise NoConstraint
        if self.verbose > 0:
            print("new slack: %f" % (slack))
        return Y_hat, dpsi, loss_mean

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
        if constraints is None:
            constraints = []
        loss_curve = []
        objective_curve = []
        self.alphas = []  # dual solutions

        # get the psi of the ground truth
        psi_gt = self.problem.batch_psi(X, Y)

        try:
            # catch ctrl+c to stop training
            for iteration in xrange(self.max_iter):
                # main loop
                if self.verbose > 0:
                    print("iteration %d" % iteration)
                    print(self)
                try:
                    Y_hat, dpsi, loss_mean = self._constraint_from_cache(
                        X, Y, w, psi_gt, constraints)
                except NoConstraint:
                    try:
                        Y_hat, dpsi, loss_mean = self._find_new_constraint(
                            X, Y, w, psi_gt, constraints)
                    except NoConstraint:
                        print("no additional constraints")
                        break

                self._compute_training_loss(X, Y, w, iteration)
                # now check the slack + the constraint
                self._update_cache(Y_hat)
                constraints.append((dpsi, loss_mean))

                w, objective = self._solve_1_slack_qp(constraints,
                                                      n_samples=len(X))
                if self.verbose > 0:
                    print("dual objective: %f" % objective)
                objective_curve.append(objective)

                if (iteration > 1 and objective_curve[-2]
                        - objective_curve[-1] < self.tol):
                    print("objective converged.")
                    break
                if self.verbose > 5:
                    print(w)
        except KeyboardInterrupt:
            pass
        self.w = w
        self.constraints_ = constraints
        print("calls to inference: %d" % self.problem.inference_calls)
        self.loss_curve_ = loss_curve
        self.objective_curve_ = objective_curve
        return self
