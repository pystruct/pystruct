import numpy as np
import matplotlib.pyplot as plt

from sklearn.externals.joblib import Parallel, delayed

from .cutting_plane_ssvm import StructuredSVM
from ..utils import find_constraint


class SubgradientStructuredSVM(StructuredSVM):
    """Structured SVM solver using subgradient descent.

    Implements a margin rescaled with l1 slack penalty.
    By default, a constant learning rate is used.
    It is also possible to use the adaptive learning rate found by AdaGrad.

    This class implements batch subgradient descent, i.e. does a single update
    for each pass over the dataset.

    Parmeters
    ---------
    problem : StructuredProblem
        Object containing problem formulation. Has to implement
        `loss`, `inference` and `loss_augmented_inference`.

    max_iter : int
        Maximum number of passes over dataset to find constraints and perform
        updates.

    C : float
        Regularization parameter

    verbose : int
        Verbosity.

    learningrate : float
        Learning rate used in subgradient descent.

    momentum : float
        Momentum used in subgradient descent.

    plot : bool (default=Fale)
        Whether to plot a learning curve in the end.

    adagrad : bool (default=False)
        Whether to use adagrad gradient scaling.
        Ignores if True, momentum is ignored.

    n_jobs : int, default=1
        Number of parallel jobs for inference. -1 means as many as cpus.

    Attributes
    ----------
    w : nd-array, shape=(problem.psi,)
        The learned weights of the SVM.

    """
    def __init__(self, problem, max_iter=100, C=1.0, verbose=0, momentum=0.9,
                 learningrate=0.001, plot=False, adagrad=False, n_jobs=1):
        StructuredSVM.__init__(self, problem, max_iter, C, verbose=verbose,
                               n_jobs=n_jobs)
        self.momentum = momentum
        self.learningrate = learningrate
        self.t = 0
        self.plot = plot
        self.adagrad = adagrad
        self.grad_old = np.zeros(self.problem.size_psi)

    def _solve_subgradient(self, psis):
        """Do a single subgradient step."""
        if hasattr(self, 'w'):
            w = self.w
        else:
            w = np.ones(self.problem.size_psi) * 1e-10
        psi_matrix = np.vstack(psis).mean(axis=0)
        #w += 1. / self.t * (psi_matrix - w / self.C / 2)
        #grad = (self.learningrate / (self.t + 1.) ** 2
                #* (psi_matrix - w / self.C / 2))
        grad = (psi_matrix - w / self.C)

        if self.adagrad:
            self.grad_old += grad ** 2
            w += self.learningrate * grad / (1. + np.sqrt(self.grad_old))
            print("grad old %f" % np.mean(self.grad_old))
            print("effective lr %f" % (self.learningrate /
                                       np.mean(1. + np.sqrt(self.grad_old))))
        else:
            grad_old = ((1 - self.momentum) * grad
                        + self.momentum * self.grad_old)
            #w += self.learningrate / (self.t + 1) * grad_old
            w += self.learningrate * grad_old

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
            verbose = max(0, self.verbose - 3)
            candidate_constraints = Parallel(n_jobs=self.n_jobs,
                                             verbose=verbose)(
                                                 delayed(find_constraint)(
                                                     self.problem, x, y, w)
                                                 for x, y in zip(X, Y))

            for i, x, y, constraint in zip(np.arange(len(X)), X, Y,
                                           candidate_constraints):
                y_hat, delta_psi, slack, loss = constraint
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
