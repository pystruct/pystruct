######################
# (c) 2012 Andreas Mueller <amueller@ais.uni-bonn.de>
# License: BSD 3-clause
#

from time import time
import numpy as np

from sklearn.externals.joblib import Parallel, delayed, cpu_count
from sklearn.utils import gen_even_slices

from .subgradient_ssvm import SubgradientSSVM
from ..utils import find_constraint_latent


class LatentSubgradientSSVM(SubgradientSSVM):
    """Latent Variable Structured SVM solver using subgradient descent.

    Implements a margin rescaled with l1 slack penalty.
    By default, a constant learning rate is used.
    It is also possible to use the adaptive learning rate found by AdaGrad.

    This class implements online subgradient descent. If n_jobs != 1,
    small batches of size n_jobs are used to exploit parallel inference.
    If inference is fast, use n_jobs=1.

    Parameters
    ----------
    model : StructuredModel
        Object containing model structure. Has to implement
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

    show_loss_every : int, default=0
        Controlls how often the hamming loss is computed (for monitoring
        purposes). Zero means never, otherwise it will be computed very
        show_loss_every'th epoch.

    decay_exponent : float, default=0
        Exponent for decaying learning rate. Effective learning rate is
        ``learning_rate / (t0 + t)** decay_exponent``. Zero means no decay.
        Ignored if adagrad=True.

    decay_t0 : float, default=10
        Offset for decaying learning rate. Effective learning rate is
        ``learning_rate / (t0 + t)** decay_exponent``. Zero means no decay.
        Ignored if adagrad=True.

    break_on_no_constraints : bool, default=True
        Break when there are no new constraints found.


    Attributes
    ----------
    w : nd-array, shape=(model.psi,)
        The learned weights of the SVM.

   ``loss_curve_`` : list of float
        List of loss values if show_loss_every > 0.

   ``objective_curve_`` : list of float
       Primal objective after each pass through the dataset.

    ``timestamps_`` : list of int
        Total training time stored before each iteration.
    """
    def __init__(self, model, max_iter=100, C=1.0, verbose=0, momentum=0.9,
                 learning_rate=0.001, adagrad=False, n_jobs=1,
                 show_loss_every=0, decay_exponent=0, decay_t0=10,
                 break_on_no_constraints=True, logger=None):
        SubgradientSSVM.__init__(
            self, model, max_iter, C, verbose=verbose, n_jobs=n_jobs,
            show_loss_every=show_loss_every, decay_exponent=decay_exponent,
            momentum=momentum, learning_rate=learning_rate, adagrad=adagrad,
            break_on_no_constraints=break_on_no_constraints, logger=logger,
            decay_t0=decay_t0)

    def fit(self, X, Y, H_init=None, warm_start=False):
        """Learn parameters using subgradient descent.

        Parameters
        ----------
        X : iterable
            Traing instances. Contains the structured input objects.
            No requirement on the particular form of entries of X is made.

        Y : iterable
            Training labels. Contains the strctured labels for inputs in X.
            Needs to have the same length as X.

        constraints : None
            Discarded. Only for API compatibility currently.

        warm_start : boolean, default=False
            Whether to restart a previous fit.
        """
        print("Training latent subgradient structural SVM")
        if not warm_start:
            self.w = getattr(self, "w", np.random.normal(
                0, 1, size=self.model.size_psi))
            self.timestamps_ = [time()]
            self.objective_curve_ = []
        else:
            # hackety hack
            self.timestamps_[0] = time() - self.timestamps_[-1]
        n_samples = len(X)
        try:
            # catch ctrl+c to stop training
            for iteration in xrange(self.max_iter):
                self.timestamps_.append(time() - self.timestamps_[0])
                positive_slacks = 0
                objective = 0.
                #verbose = max(0, self.verbose - 3)

                if self.n_jobs == 1:
                    # online learning
                    for x, y in zip(X, Y):
                        h = self.model.latent(x, y, self.w)
                        h_hat = self.model.loss_augmented_inference(
                            x, h, self.w, relaxed=True)
                        delta_psi = (self.model.psi(x, h)
                                     - self.model.psi(x, h_hat))
                        slack = (-np.dot(delta_psi, self.w)
                                 + self.model.loss(h, h_hat))
                        objective += np.maximum(slack, 0)
                        if slack > 0:
                            positive_slacks += 1
                        self._solve_subgradient(delta_psi, n_samples)
                else:
                    #generate batches of size n_jobs
                    #to speed up inference
                    if self.n_jobs == -1:
                        n_jobs = cpu_count()
                    else:
                        n_jobs = self.j_jobs

                    n_batches = int(np.ceil(float(len(X)) / n_jobs))
                    slices = gen_even_slices(n_samples, n_batches)
                    for batch in slices:
                        X_b = X[batch]
                        Y_b = Y[batch]
                        verbose = self.verbose - 1
                        candidate_constraints = Parallel(
                            n_jobs=self.n_jobs,
                            verbose=verbose)(delayed(find_constraint_latent)(
                                self.model, x, y, self.w)
                                for x, y in zip(X_b, Y_b))
                        dpsi = np.zeros(self.model.size_psi)
                        for x, y, constraint in zip(X_b, Y_b,
                                                    candidate_constraints):
                            y_hat, delta_psi, slack, loss = constraint
                            objective += slack
                            dpsi += delta_psi
                            if slack > 0:
                                positive_slacks += 1
                        dpsi /= float(len(X_b))
                        self._solve_subgradient(dpsi, n_samples)

                # some statistics
                objective *= self.C
                objective += np.sum(self.w ** 2) / 2.
                #objective /= float(n_samples)

                if positive_slacks == 0:
                    print("No additional constraints")
                    if self.break_on_no_constraints:
                        break
                if self.verbose > 0:
                    print(self)
                    print("iteration %d" % iteration)
                    print("positive slacks: %d, "
                          "objective: %f" %
                          (positive_slacks, objective))
                self.objective_curve_.append(objective)

                if self.verbose > 2:
                    print(self.w)

                self._compute_training_loss(X, Y, iteration)
                if self.logger is not None:
                    self.logger(self, iteration)

        except KeyboardInterrupt:
            pass
        print("final objective: %f" % self.objective_curve_[-1])
        if self.verbose and self.n_jobs == 1:
            print("calls to inference: %d" % self.model.inference_calls)
        return self

    def predict(self, X):
        prediction = SubgradientSSVM.predict(self, X)
        return [self.model.label_from_latent(h) for h in prediction]

    def predict_latent(self, X):
        return SubgradientSSVM.predict(self, X)

    def score(self, X, Y):
        """Compute score as 1 - loss over whole data set.

        Returns the average accuracy (in terms of model.loss)
        over X and Y.

        Parameters
        ----------
        X : iterable
            Evaluation data.

        Y : iterable
            True labels.

        Returns
        -------
        score : float
            Average of 1 - loss over training examples.
        """
        if hasattr(self.model, 'batch_loss'):
            losses = self.model.batch_loss(
                Y, self.model.batch_inference(X, self.w))
        else:
            losses = [self.model.loss(y, self.model.inference(y, self.w))
                      for y, y_pred in zip(Y, self.predict(X))]
        max_losses = [self.model.max_loss(y) for y in Y]
        return 1. - np.sum(losses) / float(np.sum(max_losses))
