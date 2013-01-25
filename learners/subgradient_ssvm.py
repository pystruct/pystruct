import numpy as np

from sklearn.externals.joblib import Parallel, delayed, cpu_count
from sklearn.utils import gen_even_slices

from .ssvm import BaseSSVM
from ..utils import find_constraint


class SubgradientStructuredSVM(BaseSSVM):
    """Structured SVM solver using subgradient descent.

    Implements a margin rescaled with l1 slack penalty.
    By default, a constant learning rate is used.
    It is also possible to use the adaptive learning rate found by AdaGrad.

    This class implements online subgradient descent. If n_jobs != 1,
    small batches of size n_jobs are used to exploit parallel inference.
    If inference is fast, use n_jobs=1.

    Parmeters
    ---------
    problem : StructuredProblem
        Object containing problem formulation. Has to implement
        `loss`, `inference` and `loss_augmented_inference`.

    max_iter : int, default=100
        Maximum number of passes over dataset to find constraints and perform
        updates.

    C : float, default=1.
        Regularization parameter

    verbose : int, default=0
        Verbosity.

    learning_rate : float, default=0.001
        Learning rate used in subgradient descent.

    momentum : float, default=0.9
        Momentum used in subgradient descent.

    adagrad : bool (default=False)
        Whether to use adagrad gradient scaling.
        Ignores if True, momentum is ignored.

    n_jobs : int, default=1
        Number of parallel jobs for inference. -1 means as many as cpus.

    show_loss : string, default='augmented'
        Controlls the meaning of the loss curve and convergence messages.
        By default (show_loss='augmented') the loss of the loss-augmented
        prediction is shown, since this is computed any way.
        Setting show_loss='true' will show the true loss, i.e. the one of
        the normal prediction. Be aware that this means an additional
        call to inference in each iteration!

    Attributes
    ----------
    w : nd-array, shape=(problem.psi,)
        The learned weights of the SVM.

   ``loss_curve_`` : list of float
        List of loss values after each pass thorugh the dataset.
        Either sum of slacks (loss on loss augmented predictions)
        or actual loss, depending on the value of ``show_loss``.

   ``objective_curve_`` : list of float
       Primal objective after each pass through the dataset.

    """
    def __init__(self, problem, max_iter=100, C=1.0, verbose=0, momentum=0.9,
                 learning_rate=0.001, adagrad=False, n_jobs=1,
                 show_loss='augmented'):
        BaseSSVM.__init__(self, problem, max_iter, C, verbose=verbose,
                          n_jobs=n_jobs, show_loss=show_loss)
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.t = 0
        self.adagrad = adagrad
        self.grad_old = np.zeros(self.problem.size_psi)

    def _solve_subgradient(self, w, dpsi):
        """Do a single subgradient step."""

        #w += 1. / self.t * (psi_matrix - w / self.C / 2)
        #grad = (self.learning_rate / (self.t + 1.) ** 2
                #* (psi_matrix - w / self.C / 2))
        grad = (dpsi - w / self.C)

        if self.adagrad:
            self.grad_old += grad ** 2
            w += self.learning_rate * grad / (1. + np.sqrt(self.grad_old))
            print("grad old %f" % np.mean(self.grad_old))
            print("effective lr %f" % (self.learning_rate /
                                       np.mean(1. + np.sqrt(self.grad_old))))
        else:
            self.grad_old = ((1 - self.momentum) * grad
                             + self.momentum * self.grad_old)
            #w += self.learning_rate / (self.t + 1) * grad_old
            w += self.learning_rate * self.grad_old

        self.w = w
        self.t += 1.
        return w

    def fit(self, X, Y):
        """Learn parameters using subgradient descent.

        Parameters
        ----------
        X : iterable
            Traing instances. Contains the structured input objects.
            No requirement on the particular form of entries of X is made.

        Y : iterable
            Training labels. Contains the strctured labels for inputs in X.
            Needs to have the same length as X.
        """
        print("Training primal subgradient structural SVM")
        w = getattr(self, "w", np.zeros(self.problem.size_psi))
        #constraints = []
        loss_curve = []
        objective_curve = []
        n_samples = len(X)
        try:
            # catch ctrl+c to stop training
            for iteration in xrange(self.max_iter):
                positive_slacks = 0
                current_loss = 0.
                objective = 0.
                verbose = max(0, self.verbose - 3)

                if self.n_jobs == 1:
                    # online learning
                    for x, y in zip(X, Y):
                        y_hat, delta_psi, slack, loss = \
                            find_constraint(self.problem, x, y, w)
                        objective += slack
                        current_loss += self._get_loss(x, y, w, loss)
                        if slack > 0:
                            positive_slacks += 1
                        w = self._solve_subgradient(w, delta_psi)
                else:
                    # generate batches of size n_jobs
                    # to speed up inference
                    if self.n_jobs == -1:
                        n_jobs = cpu_count()
                    else:
                        n_jobs = self.j_jobs

                    n_batches = int(np.ceil(float(len(X)) / n_jobs))
                    slices = gen_even_slices(n_samples, n_batches)
                    for batch in slices:
                        X_b = X[batch]
                        Y_b = Y[batch]
                        candidate_constraints = Parallel(
                            n_jobs=self.n_jobs,
                            verbose=verbose)(delayed(find_constraint)(
                                self.problem, x, y, w)
                                for x, y in zip(X_b, Y_b))
                        dpsi = np.zeros(self.problem.size_psi)
                        for x, y, constraint in zip(X_b, Y_b,
                                                    candidate_constraints):
                            y_hat, delta_psi, slack, loss = constraint
                            objective += slack
                            dpsi += delta_psi
                            current_loss += self._get_loss(x, y, w, loss)
                            if slack > 0:
                                positive_slacks += 1
                        dpsi /= float(len(X_b))
                        w = self._solve_subgradient(w, dpsi)

                # some statistics
                objective /= len(X)
                current_loss /= len(X)
                objective += np.sum(w ** 2) / self.C / 2.
                if positive_slacks == 0:
                    print("No additional constraints")
                    break
                if self.verbose > 0:
                    print("iteration %d" % iteration)
                    print("current loss: %f  positive slacks: %d,"
                          "objective: %f" %
                          (current_loss, positive_slacks, objective))
                loss_curve.append(current_loss)
                objective_curve.append(objective)

                if self.verbose > 2:
                    print(w)
        except KeyboardInterrupt:
            pass
        self.w = w
        self.loss_curve_ = loss_curve
        self.objective_curve_ = objective_curve
        print("final objective: %f" % objective_curve[-1])
        print("calls to inference: %d" % self.problem.inference_calls)
