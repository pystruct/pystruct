######################
# (c) 2012 Andreas Mueller <amueller@ais.uni-bonn.de>
# License: BSD 3-clause
#
# Implements structured SVM as described in Tsochantaridis et. al.
# Support Vector Machines Learning for Interdependent
# and Structures Output Spaces

from time import time

import numpy as np
import cvxopt
import cvxopt.solvers

from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import gen_even_slices

from .ssvm import BaseSSVM
from ..utils import unwrap_pairwise, find_constraint


class NSlackSSVM(BaseSSVM):
    """Structured SVM solver for the n-slack QP with l1 slack penalty.

    Implements margin rescaled structural SVM using
    the n-slack formulation and cutting plane method, solved using CVXOPT.
    The optimization is restarted in each iteration.

    Parameters
    ----------
    model : StructuredModel
        Object containing the model structure. Has to implement
        `loss`, `inference` and `loss_augmented_inference`.

    max_iter : int
        Maximum number of passes over dataset to find constraints.

    C : float
        Regularization parameter

    check_constraints : bool (default=True)
        Whether to check if the new "most violated constraint" is
        more violated than previous constraints. Helpful for stopping
        and debugging, but costly.

    verbose : int (default=0)
        Verbosity.

    negativity_constraint: list of ints
        Indices of parmeters that are constraint to be negative.
        This is useful for learning submodular CRFs (inference is formulated
        as maximization in SSVMs, flipping some signs).

    break_on_bad: bool (default=False)
        Whether to break (start debug mode) when inference was approximate.

    n_jobs : int, default=1
        Number of parallel jobs for inference. -1 means as many as cpus.

    show_loss_every : int, default=0
        Controlls how often the hamming loss is computed (for monitoring
        purposes). Zero means never, otherwise it will be computed very
        show_loss_every'th epoch.

    batch_size : int, default=100
        Number of constraints after which we solve the QP again.
        batch_size=-1 means that an update is performed only after going once
        over the whole training set.

    tol : float, default=-10
        Convergence tolerance. If dual objective decreases less than tol,
        learning is stopped. The default corresponds to ignoring the behavior
        of the dual objective and stop only if no more constraints can be
        found.

    inactive_threshold : float, default=1e-5
        Threshold for dual variable of a constraint to be considered inactive.

    inactive_window : float, default=50
        Window for measuring inactivity. If a constraint is inactive for
        ``inactive_window`` iterations, it will be pruned from the QP.
        If set to 0, no constraints will be removed.

    switch_to : None or string, default=None
        Switch to the given inference method if the previous method does not
        find any more constraints.

    logger : logger object, default=None
        Pystruct logger for storing the model or extracting additional
        information.

    Attributes
    ----------
    w : nd-array, shape=(model.size_joint_feature,)
        The learned weights of the SVM.

    old_solution : dict
        The last solution found by the qp solver.

    ``loss_curve_`` : list of float
        List of loss values if show_loss_every > 0.

    ``objective_curve_`` : list of float
        Cutting plane objective after each pass through the dataset.

    ``primal_objective_curve_`` : list of float
        Primal objective after each pass through the dataset.

    ``timestamps_`` : list of int
       Total training time stored before each iteration.

    References
    ----------
    * Tsochantaridis, Ioannis and Joachims, Thorsten and Hofmann, Thomas and
        Altun, Yasemin and Singer, Yoram: Large margin methods for structured
        and interdependent output variables, JMLR 2006

    * Joachims, Thorsten and Finley, Thomas and Yu, Chun-Nam John:
        Cutting-plane training of structural SVMs, JMLR 2009
    """

    def __init__(self, model, max_iter=100, C=1.0, check_constraints=True,
                 verbose=0, negativity_constraint=None, n_jobs=1,
                 break_on_bad=False, show_loss_every=0, batch_size=100,
                 tol=1e-3, inactive_threshold=1e-5,
                 inactive_window=50, logger=None, switch_to=None):

        BaseSSVM.__init__(self, model, max_iter, C, verbose=verbose,
                          n_jobs=n_jobs, show_loss_every=show_loss_every,
                          logger=logger)

        self.negativity_constraint = negativity_constraint
        self.check_constraints = check_constraints
        self.break_on_bad = break_on_bad
        self.batch_size = batch_size
        self.tol = tol
        self.inactive_threshold = inactive_threshold
        self.inactive_window = inactive_window
        self.switch_to = switch_to

    def _solve_n_slack_qp(self, constraints, n_samples):
        C = self.C
        joint_features = [c[1] for sample in constraints for c in sample]
        losses = [c[2] for sample in constraints for c in sample]

        joint_feature_matrix = np.vstack(joint_features).astype(np.float)
        n_constraints = len(joint_features)
        P = cvxopt.matrix(np.dot(joint_feature_matrix, joint_feature_matrix.T))
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
        if self.negativity_constraint is None:
            #empty constraints
            zero_constr = np.zeros(0)
            joint_features_constr = np.zeros((0, n_constraints))
        else:
            joint_features_constr = joint_feature_matrix.T[self.negativity_constraint]
            zero_constr = np.zeros(len(self.negativity_constraint))

        # put together
        G = cvxopt.sparse(cvxopt.matrix(np.vstack((-idy, blocks,
                                                   joint_features_constr))))
        tmp2 = np.ones(n_samples) * C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2, zero_constr)))

        # solve QP model
        cvxopt.solvers.options['feastol'] = 1e-5
        try:
            solution = cvxopt.solvers.qp(P, q, G, h)
        except ValueError:
            solution = {'status': 'error'}
        if solution['status'] != "optimal":
            print("regularizing QP!")
            P = cvxopt.matrix(np.dot(joint_feature_matrix, joint_feature_matrix.T)
                              + 1e-8 * np.eye(joint_feature_matrix.shape[0]))
            solution = cvxopt.solvers.qp(P, q, G, h)
            if solution['status'] != "optimal":
                raise ValueError("QP solver failed. Try regularizing your QP.")

        # Lagrange multipliers
        a = np.ravel(solution['x'])
        self.prune_constraints(constraints, a)
        self.old_solution = solution

        # Support vectors have non zero lagrange multipliers
        sv = a > self.inactive_threshold * C
        box = np.dot(blocks, a)
        if self.verbose > 1:
            print("%d support vectors out of %d points" % (np.sum(sv),
                                                           n_constraints))
            # calculate per example box constraint:
            print("Box constraints at C: %d" % np.sum(1 - box / C < 1e-3))
            print("dual objective: %f" % -solution['primal objective'])
        self.w = np.dot(a, joint_feature_matrix)
        return -solution['primal objective']

    def _check_bad_constraint(self, y_hat, slack, old_constraints):
        if slack < 1e-5:
            return True
        y_hat_plain = unwrap_pairwise(y_hat)

        already_active = np.any([True for y__, _, _ in old_constraints
                                 if np.all(y_hat_plain ==
                                           unwrap_pairwise(y__))])
        if already_active:
            return True

        # "smart" stopping criterion
        # check if most violated constraint is more violated
        # than previous ones by more then eps.
        # If it is less violated, inference was wrong/approximate
        if self.check_constraints:
            for con in old_constraints:
                # compute slack for old constraint
                slack_tmp = max(con[2] - np.dot(self.w, con[1]), 0)
                if self.verbose > 5:
                    print("slack old constraint: %f" % slack_tmp)
                # if slack of new constraint is smaller or not
                # significantly larger, don't add constraint.
                # if smaller, complain about approximate inference.
                if slack - slack_tmp < -1e-5:
                    if self.verbose > 0:
                        print("bad inference: %f" % (slack_tmp - slack))
                    if self.break_on_bad:
                        raise ValueError("bad inference: %f" % (slack_tmp -
                                                                slack))
                    return True

        return False

    def fit(self, X, Y, constraints=None, warm_start=None, initialize=True):
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
            Each constraint is of the form [y_hat, delta_joint_feature, loss], where
            y_hat is a labeling, ``delta_joint_feature = joint_feature(x, y) - joint_feature(x, y_hat)``
            and loss is the loss for predicting y_hat instead of the true label
            y.

        initialize : boolean, default=True
            Whether to initialize the model for the data.
            Leave this true except if you really know what you are doing.
        """
        if self.verbose:
            print("Training n-slack dual structural SVM")
        cvxopt.solvers.options['show_progress'] = self.verbose > 3
        if initialize:
            self.model.initialize(X, Y)
        self.w = np.zeros(self.model.size_joint_feature)
        n_samples = len(X)
        stopping_criterion = False
        if constraints is None:
            # fresh start
            constraints = [[] for i in range(n_samples)]
            self.last_active = [[] for i in range(n_samples)]
            self.objective_curve_ = []
            self.primal_objective_curve_ = []
            self.timestamps_ = [time()]
        else:
            # warm start
            objective = self._solve_n_slack_qp(constraints, n_samples)
        try:
            # catch ctrl+c to stop training
            # we have to update at least once after going through the dataset
            for iteration in range(self.max_iter):
                # main loop
                self.timestamps_.append(time() - self.timestamps_[0])
                if self.verbose > 0:
                    print("iteration %d" % iteration)
                if self.verbose > 2:
                    print(self)
                new_constraints = 0
                # generate slices through dataset from batch_size
                if self.batch_size < 1 and not self.batch_size == -1:
                    raise ValueError("batch_size should be integer >= 1 or -1,"
                                     "got %s." % str(self.batch_size))
                batch_size = (self.batch_size if self.batch_size != -1 else
                              len(X))
                n_batches = int(np.ceil(float(len(X)) / batch_size))
                slices = gen_even_slices(n_samples, n_batches)
                indices = np.arange(n_samples)
                slack_sum = 0
                for batch in slices:
                    new_constraints_batch = 0
                    verbose = max(0, self.verbose - 3)
                    X_b = X[batch]
                    Y_b = Y[batch]
                    indices_b = indices[batch]
                    candidate_constraints = Parallel(
                        n_jobs=self.n_jobs, verbose=verbose)(
                            delayed(find_constraint)(self.model, x, y, self.w)
                            for x, y in zip(X_b, Y_b))

                    # for each batch, gather new constraints
                    for i, x, y, constraint in zip(indices_b, X_b, Y_b,
                                                   candidate_constraints):
                        # loop over samples in batch
                        y_hat, delta_joint_feature, slack, loss = constraint
                        slack_sum += slack

                        if self.verbose > 3:
                            print("current slack: %f" % slack)

                        if not loss > 0:
                            # can have y != y_hat but loss = 0 in latent svm.
                            # we need this here as djoint_feature is then != 0
                            continue

                        if self._check_bad_constraint(y_hat, slack,
                                                      constraints[i]):
                            continue

                        constraints[i].append([y_hat, delta_joint_feature, loss])
                        new_constraints_batch += 1

                    # after processing the slice, solve the qp
                    if new_constraints_batch:
                        objective = self._solve_n_slack_qp(constraints,
                                                           n_samples)
                        new_constraints += new_constraints_batch

                self.objective_curve_.append(objective)
                self._compute_training_loss(X, Y, iteration)

                primal_objective = (self.C
                                    * slack_sum
                                    + np.sum(self.w ** 2) / 2)
                self.primal_objective_curve_.append(primal_objective)

                if self.verbose > 0:
                    print("new constraints: %d, "
                          "cutting plane objective: %f primal objective: %f" %
                          (new_constraints, objective, primal_objective))

                if new_constraints == 0:
                    if self.verbose:
                        print("no additional constraints")
                    stopping_criterion = True

                if (iteration > 1 and self.objective_curve_[-1]
                        - self.objective_curve_[-2] < self.tol):
                    if self.verbose:
                        print("objective converged.")
                    stopping_criterion = True

                if stopping_criterion:
                    if (self.switch_to is not None and
                            self.model.inference_method != self.switch_to):
                        if self.verbose:
                            print("Switching to %s inference" %
                                  str(self.switch_to))
                        self.model.inference_method_ = \
                            self.model.inference_method
                        self.model.inference_method = self.switch_to
                        stopping_criterion = False
                        continue
                    else:
                        break

                if self.verbose > 5:
                    print(self.w)

                if self.logger is not None:
                    self.logger(self, iteration)
        except KeyboardInterrupt:
            pass

        self.constraints_ = constraints
        if self.verbose and self.n_jobs == 1:
            print("calls to inference: %d" % self.model.inference_calls)

        if verbose:
            print("Computing final objective.")
        self.timestamps_.append(time() - self.timestamps_[0])
        self.primal_objective_curve_.append(self._objective(X, Y))
        self.objective_curve_.append(objective)
        if self.logger is not None:
            self.logger(self, 'final')
        return self

    def prune_constraints(self, constraints, a):
        # append list for new constraint
        # self.alpha is a list which has
        # an entry per sample. each sample has an int for each constraint,
        # saying when was it last used
        if self.inactive_window == 0:
            return
        k = 0
        for i, sample in enumerate(constraints):
            # if there are no constraints for this sample, do nothing:
            if not len(sample):
                continue
            # add self.last_active for any new constraint
            n_old_constraints_sample = len(self.last_active[i])
            if n_old_constraints_sample < len(sample):
                self.last_active[i] = np.hstack([self.last_active[i], [0]])
            # if inactive, count up
            inactive_this = (a[k:k + len(sample)] < self.inactive_threshold
                             * self.C)
            self.last_active[i][inactive_this] += 1
            k += len(sample)
            assert(len(sample) == len(self.last_active[i]))

            # remove unused constraints:
            to_remove = self.last_active[i] > self.inactive_window
            self.last_active[i] = self.last_active[i][~to_remove]
            for j in np.where(to_remove)[0][::-1]:
                del sample[j]
            assert(len(sample) == len(self.last_active[i]))
