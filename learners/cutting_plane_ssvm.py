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

from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import gen_even_slices

from .ssvm import BaseSSVM
from ..utils import unwrap_pairwise, find_constraint


class StructuredSVM(BaseSSVM):
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
        batch_size=-1 means that an update is performed only after going once
        over the whole training set.


    Attributes
    ----------
    w : nd-array, shape=(problem.psi,)
        The learned weights of the SVM.

    old_solution : dict
        The last solution found by the qp solver.

   ``loss_curve_`` : list of float
        List of loss values after each pass thorugh the dataset.
        Either sum of slacks (loss on loss augmented predictions)
        or actual loss, depending on the value of ``show_loss``.

   ``objective_curve_`` : list of float
       Primal objective after each pass through the dataset.
    """

    def __init__(self, problem, max_iter=100, C=1.0, check_constraints=True,
                 verbose=1, positive_constraint=None, n_jobs=1,
                 break_on_bad=True, show_loss='true', batch_size=100,
                 tol=0.0001):

        BaseSSVM.__init__(self, problem, max_iter, C, verbose=verbose,
                          n_jobs=n_jobs, show_loss=show_loss)

        self.positive_constraint = positive_constraint
        self.check_constraints = check_constraints
        self.break_on_bad = break_on_bad
        self.batch_size = batch_size
        self.tol = tol
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
                from IPython.core.debugger import Tracer
                Tracer()()

        # Lagrange multipliers
        a = np.ravel(solution['x'])
        self.alphas.append(a)
        self.old_solution = solution

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-10
        box = np.dot(blocks, a)
        if self.verbose > 1:
            print("%d support vectors out of %d points" % (np.sum(sv),
                                                           n_constraints))
            # calculate per example box constraint:
            print("Box constraints at C: %d" % np.sum(1 - box / C < 1e-3))
            print("dual objective: %f" % solution['primal objective'])
        w = np.dot(a, psi_matrix)
        return w, solution['primal objective']

    def _check_bad_constraint(self, y_hat, slack, old_constraints, w):
        if slack < 1e-5:
            return True
        y_hat_plain = unwrap_pairwise(y_hat)

        already_active = np.any([True for y__, _, _ in old_constraints
                                 if (y_hat_plain ==
                                     unwrap_pairwise(y__)).all()])
        if already_active:
            return True

        # "smart" stopping criterion
        # check if most violated constraint is more violated
        # than previous ones by more then eps.
        # If it is less violated, inference was wrong/approximate
        if self.check_constraints:
            for con in old_constraints:
                # compute slack for old constraint
                slack_tmp = max(con[2] - np.dot(w, con[1]), 0)
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
        print("Training n-slack dual structural SVM")
        w = np.zeros(self.problem.size_psi)
        n_samples = len(X)
        if constraints is None:
            constraints = [[] for i in xrange(n_samples)]
            new_constraints = 0
        else:
            new_constraints = np.sum([len(c) for c in constraints])
        loss_curve = []
        objective_curve = []
        self.alphas = []  # dual solutions
        # we have to update at least once after going through the dataset
        for iteration in xrange(self.max_iter):
            # main loop
            if self.verbose > 0:
                print("iteration %d" % iteration)
            if iteration > 0:
                # don't reset in the first iteration
                # we might have been passed some
                new_constraints = 0
            current_loss = 0.
            # generate slices through dataset from batch_size
            n_batches = int(np.ceil(float(len(X)) / self.batch_size))
            slices = gen_even_slices(n_samples, n_batches)
            for batch in slices:
                verbose = max(0, self.verbose - 3)
                X_b = X[batch]
                Y_b = Y[batch]
                candidate_constraints = Parallel(n_jobs=self.n_jobs,
                                                 verbose=verbose)(
                                                     delayed(find_constraint)(
                                                         self.problem, x, y, w)
                                                     for x, y in zip(X_b, Y_b))

                # for each slice, gather new constraints
                for i, x, y, constraint in zip(np.arange(len(X_b)), X_b, Y_b,
                                               candidate_constraints):
                    # loop over dataset
                    y_hat, delta_psi, slack, loss = constraint

                    current_loss += self._get_loss(x, y, w, loss)
                    if self.verbose > 3:
                        print("current slack: %f" % slack)

                    if not loss > 0:
                        # can have y != y_hat but loss = 0 in latent svm.
                        # we need this here as dpsi is then != 0
                        continue

                    if self._check_bad_constraint(y_hat, slack, constraints[i],
                                                  w):
                        continue

                    constraints[i].append([y_hat, delta_psi, loss])
                    new_constraints += 1
                # after processing the slice, solve the qp
                w, objective = self._solve_n_slack_qp(constraints,
                                                      n_samples)
                objective_curve.append(objective)

            if new_constraints == 0:
                print("no additional constraints")
                break

            current_loss /= len(X)
            loss_curve.append(current_loss)

            if self.verbose > 0:
                print("current loss: %f  new constraints: %d, "
                      "dual objective: %f" %
                      (current_loss, new_constraints,
                       objective))
            if (iteration > 1 and objective_curve[-2]
                    - objective_curve[-1] < self.tol):
                print("objective converged.")
                break
            if self.verbose > 5:
                print(w)

        self.w = w
        self.constraints_ = constraints
        self.loss_curve_ = loss_curve
        self.objective_curve_ = objective_curve
        print("calls to inference: %d" % self.problem.inference_calls)
