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

    tol : float, default=1e-5
        Convergence tolerance. If dual objective decreases less than tol,
        learning is stopped. The default corresponds to ignoring the behavior
        of the dual objective and stop only if no more constraints can be
        found.

    inference_cache : int, default=0
        How many results of loss_augmented_inference to cache per sample.
        If > 0 the most violating of the cached examples will be used to
        construct a global constraint. Only if this constraint is not violated,
        inference will be run again. This parameter poses a memory /
        computation tradeoff. Storing more constraints might lead to RAM being
        exhausted. Using inference_cache > 0 is only advisable if computation
        time is dominated by inference.

    cache_tol : float, default=None
        Tolerance when to reject a constraint from cache (and do inference).
        If None, ``tol`` will be used. Higher values might lead to faster
        learning.

    inactive_threshold : float, default=1e-5
        Threshold for dual variable of a constraint to be considered inactive.

    inactive_window : float, default=50
        Window for measuring inactivity. If a constraint is inactive for
        ``inactive_window`` iterations, it will be pruned from the QP.
        If set to 0, no constraints will be removed.

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
                 break_on_bad=True, show_loss_every=0, tol=1e-5,
                 inference_cache=0, inactive_threshold=1e-10,
                 inactive_window=50, logger=None, cache_tol='auto'):

        BaseSSVM.__init__(self, problem, max_iter, C, verbose=verbose,
                          n_jobs=n_jobs, show_loss_every=show_loss_every,
                          logger=logger)

        self.positive_constraint = positive_constraint
        self.check_constraints = check_constraints
        self.break_on_bad = break_on_bad
        self.tol = tol
        self.cache_tol = cache_tol
        self.inference_cache = inference_cache
        self.inactive_threshold = inactive_threshold
        self.inactive_window = inactive_window

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
        #if hasattr(self, 'old_solution'):
            #s = self.old_solution['s']
            ## put s slightly inside the cone..
            #s = cvxopt.matrix(np.vstack([s, [[1e-10]]]))
            #z = self.old_solution['z']
            #z = cvxopt.matrix(np.vstack([z, [[1e-10]]]))
            #initvals = {'x': self.old_solution['x'], 'y':
                        #self.old_solution['y'], 'z': z,
                        #'s': s}
        #else:
            #initvals = {}
        #solution = cvxopt.solvers.qp(P, q, G, h, A, b, initvals=initvals)
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
        self.old_solution = solution
        self.prune_constraints(constraints, a)

        # Support vectors have non zero lagrange multipliers
        sv = a > self.inactive_threshold * C
        if self.verbose > 1:
            print("%d support vectors out of %d points" % (np.sum(sv),
                                                           n_constraints))
        self.w = np.dot(a, psi_matrix)
        # we needed to flip the sign to make the dual into a minimization
        # problem
        return -solution['primal objective']

    def prune_constraints(self, constraints, a):
        # append list for new constraint
        self.alphas.append([])
        assert(len(self.alphas) == len(constraints))
        for constraint, alpha in zip(self.alphas, a):
            constraint.append(alpha)
            constraint = constraint[-self.inactive_window:]

        # prune unused constraints:
        # if the max of alpha in last 50 iterations was small, throw away
        if self.inactive_window != 0:
            max_active = [np.max(constr[-self.inactive_window:])
                          for constr in self.alphas]
            # find strongest constraint that is not ground truth constraint
            strongest = np.max(max_active[1:])
            inactive = np.where(max_active
                                < self.inactive_threshold * strongest)[0]

            for idx in reversed(inactive):
                # if we don't reverse, we'll mess the indices up
                del constraints[idx]
                del self.alphas[idx]

    def _check_bad_constraint(self, violation, dpsi_mean, loss,
                              old_constraints, break_on_bad, tol=None):
        violation_difference = violation - self.last_slack_
        if self.verbose > 1:
            print("New violation: %f difference to last: %f"
                  % (violation, violation_difference))
        if violation_difference < 0 and violation > 0 and break_on_bad:
            from IPython.core.debugger import Tracer
            Tracer()()
        if tol is None:
            tol = self.tol
        if (violation_difference) < tol:
            print("new constraint to weak.")
            return True
        equals = [True for dpsi_, loss_ in old_constraints
                  if (np.all(dpsi_ == dpsi_mean) and loss == loss_)]

        if np.any(equals):
            return True

        if self.check_constraints:
            for con in old_constraints:
                # compute violation for old constraint
                violation_tmp = max(con[1] - np.dot(self.w, con[0]), 0)
                if self.verbose > 5:
                    print("violation old constraint: %f" % violation_tmp)
                # if violation of new constraint is smaller or not
                # significantly larger, don't add constraint.
                # if smaller, complain about approximate inference.
                if violation - violation_tmp < -1e-5:
                    print("bad inference: %f" % (violation_tmp - violation))
                    if break_on_bad:
                        from IPython.core.debugger import Tracer
                        Tracer()()
                    return True
        return False

    def _update_cache(self, X, Y, Y_hat):
        """Updated cached constraints."""
        if self.inference_cache == 0:
            return
        if (not hasattr(self, "inference_cache_")
                or self.inference_cache_ is None):
            self.inference_cache_ = [[] for y in Y_hat]

        def constraint_equal(y_1, y_2):
            if isinstance(y_1, tuple):
                return np.all(y_1[0] == y_2[1]) and np.all(y_1[1] == y_2[1])
            return np.all(y_1 == y_2)

        for sample, x, y, y_hat in zip(self.inference_cache_, X, Y, Y_hat):
            already_there = [constraint_equal(y_hat, cache[2])
                             for cache in sample]
            if np.any(already_there):
                continue
            if len(sample) > self.inference_cache:
                sample.pop(0)
            # we computed both of these before, but summed them up immediately
            # this makes it a little less efficient in the caching case.
            # the idea is that if we cache, inference is way more expensive
            # and this doesn't matter much.
            sample.append((self.problem.psi(x, y_hat),
                           self.problem.loss(y, y_hat), y_hat))

    def _constraint_from_cache(self, X, Y, psi_gt, constraints):
        if (not getattr(self, 'inference_cache_', False) or
                self.inference_cache_ is False):
            if self.verbose > 10:
                print("Empty cache.")
            raise NoConstraint
        if (self.cache_tol == 'auto' and
                (self.primal_objective_curve_[-1] - self.objective_curve_[-1])
                < self.cache_tol_):
            # do inference if gap has become to small
            if self.verbose > 1:
                print("Last gap too small, not loading constraint from cache.")
            raise NoConstraint

        Y_hat = []
        psi_acc = np.zeros(self.problem.size_psi)
        loss_mean = 0
        for cached in self.inference_cache_:
            # cached has entries of form (psi, loss, y_hat)
            violations = [np.dot(psi, self.w) + loss
                          for psi, loss, _ in cached]
            psi, loss, y_hat = cached[np.argmax(violations)]
            Y_hat.append(y_hat)
            psi_acc += psi
            loss_mean += loss

        dpsi = (psi_gt - psi_acc) / len(X)
        loss_mean = loss_mean / len(X)

        violation = loss_mean - np.dot(self.w, dpsi)
        if self._check_bad_constraint(violation, dpsi, loss_mean, constraints,
                                      break_on_bad=False, tol=self.cache_tol_):
            if self.verbose > 1:
                print("No constraint from cache.")
            raise NoConstraint
        return Y_hat, dpsi, loss_mean

    def _find_new_constraint(self, X, Y, psi_gt, constraints, check=True):
        if self.n_jobs != 1:
            # do inference in parallel
            verbose = max(0, self.verbose - 3)
            Y_hat = Parallel(n_jobs=self.n_jobs, verbose=verbose)(
                delayed(loss_augmented_inference)(
                    self.problem, x, y, self.w, relaxed=True)
                for x, y in zip(X, Y))
        else:
            Y_hat = self.problem.batch_loss_augmented_inference(
                X, Y, self.w, relaxed=True)
        # compute the mean over psis and losses

        dpsi = (psi_gt - self.problem.batch_psi(X, Y_hat)) / len(X)
        loss_mean = np.mean(self.problem.batch_loss(Y, Y_hat))

        violation = loss_mean - np.dot(self.w, dpsi)
        if check and self._check_bad_constraint(
                violation, dpsi, loss_mean, constraints,
                break_on_bad=self.break_on_bad):
            raise NoConstraint
        return Y_hat, dpsi, loss_mean

    def fit(self, X, Y, constraints=None, warm_start=False):
        """Learn parameters using cutting plane method.

        Parameters
        ----------
        X : iterable
            Traing instances. Contains the structured input objects.
            No requirement on the particular form of entries of X is made.

        Y : iterable
            Training labels. Contains the strctured labels for inputs in X.
            Needs to have the same length as X.

        contraints : ignored

        warm_start : bool, default=False
            Whether we are warmstarting from a previous fit.
        """
        print("Training 1-slack dual structural SVM")
        if self.verbose < 2:
            cvxopt.solvers.options['show_progress'] = False
        else:
            cvxopt.solvers.options['show_progress'] = True

        # parse cache_tol parameter
        if self.cache_tol is None or self.cache_tol == 'auto':
            self.cache_tol_ = self.tol
        else:
            self.cache_tol_ = self.cache_tol

        if not warm_start:
            self.w = np.zeros(self.problem.size_psi)
            constraints = []
            self.objective_curve_, self.primal_objective_curve_ = [], []
            self.cached_constraint_ = []
            self.alphas = []  # dual solutions
            self.last_slack_ = -1
            # append constraint given by ground truth to make our life easier
            constraints.append((np.zeros(self.problem.size_psi), 0))
            self.alphas.append([self.C])
            self.inference_cache_ = None
        else:
            constraints = self.constraints_

        # get the psi of the ground truth
        psi_gt = self.problem.batch_psi(X, Y)

        try:
            # catch ctrl+c to stop training

            for iteration in xrange(self.max_iter):
                # main loop
                cached_constraint = False
                if self.verbose > 0:
                    print("iteration %d" % iteration)
                    print(self)
                try:
                    Y_hat, dpsi, loss_mean = self._constraint_from_cache(
                        X, Y, psi_gt, constraints)
                    cached_constraint = True
                except NoConstraint:
                    try:
                        Y_hat, dpsi, loss_mean = self._find_new_constraint(
                            X, Y, psi_gt, constraints)
                        self._update_cache(X, Y, Y_hat)
                    except NoConstraint:
                        print("no additional constraints")
                        break

                self._compute_training_loss(X, Y, iteration)
                constraints.append((dpsi, loss_mean))

                # compute primal objective
                last_slack = -np.dot(self.w, dpsi) + loss_mean
                primal_objective = (self.C * len(X)
                                    * np.max(last_slack, 0)
                                    + np.sum(self.w ** 2) / 2)
                self.primal_objective_curve_.append(primal_objective)
                self.cached_constraint_.append(cached_constraint)

                objective = self._solve_1_slack_qp(constraints,
                                                   n_samples=len(X))

                # update cache tolerance if cache_tol is auto:
                if self.cache_tol == "auto" and not cached_constraint:
                    self.cache_tol_ = (primal_objective - objective) / 4.

                self.last_slack_ = np.max([(-np.dot(self.w, dpsi) + loss_mean)
                                           for dpsi, loss_mean in constraints])
                self.last_slack_ = max(self.last_slack_, 0)

                if self.verbose > 0:
                    cutting_plane_objective = (self.C * len(X)
                                               * self.last_slack_
                                               + np.sum(self.w ** 2) / 2)
                    print("dual objective: %f, cutting plane objective: %f,"
                          " primal objective %f" % (objective,
                          cutting_plane_objective, primal_objective))
                    if (np.abs(cutting_plane_objective - objective)
                            / max(np.abs(objective), 1) > .1):
                        from IPython.core.debugger import Tracer
                        Tracer()()
                # we only do this here because we didn't add the gt to the
                # constraints, which makes the dual behave a bit oddly
                self.objective_curve_.append(objective)
                self.constraints_ = constraints
                if self.logger is not None:
                    self.logger(self, iteration)

                if self.verbose > 5:
                    print(self.w)
        except KeyboardInterrupt:
            pass
        if self.logger is not None:
            self.logger(self, 'final')
        print("calls to inference: %d" % self.problem.inference_calls)
        # compute final objective:
        Y_hat, dpsi, loss_mean = self._find_new_constraint(
            X, Y, psi_gt, constraints, check=False)
        last_slack = -np.dot(self.w, dpsi) + loss_mean
        primal_objective = (self.C * len(X)
                            * np.max(last_slack, 0)
                            + np.sum(self.w ** 2) / 2)
        self.primal_objective_curve_.append(primal_objective)
        if self.verbose > 0:
            print("final primal objective: %f gap: %f"
                  % (primal_objective, primal_objective - objective))
        return self
