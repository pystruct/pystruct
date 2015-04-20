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


class SubgradientLatentSSVM(SubgradientSSVM):
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

    learning_rate : float or 'auto', default='auto'
        Learning rate used in subgradient descent. If 'auto', the pegasos
        schedule is used, which starts with ``learning_rate = n_samples * C``.

    momentum : float, default=0.0
        Momentum used in subgradient descent.

    n_jobs : int, default=1
        Number of parallel jobs for inference. -1 means as many as cpus.

    show_loss_every : int, default=0
        Controlls how often the hamming loss is computed (for monitoring
        purposes). Zero means never, otherwise it will be computed very
        show_loss_every'th epoch.

    decay_exponent : float, default=1
        Exponent for decaying learning rate. Effective learning rate is
        ``learning_rate / (decay_t0 + t)** decay_exponent``. Zero means no decay.

    decay_t0 : float, default=10
        Offset for decaying learning rate. Effective learning rate is
        ``learning_rate / (decay_t0 + t)** decay_exponent``.

    break_on_no_constraints : bool, default=True
        Break when there are no new constraints found.

    averaging : string, default=None
        Whether and how to average weights. Possible options are 'linear', 'squared' and None.
        The string reflects the weighting of the averaging:

            - linear: ``w_avg ~ w_1 + 2 * w_2 + ... + t * w_t``

            - squared: ``w_avg ~ w_1 + 4 * w_2 + ... + t**2 * w_t``

        Uniform averaging is not implemented as it is worse than linear
        weighted averaging or no averaging.



    Attributes
    ----------
    w : nd-array, shape=(model.size_joint_feature,)
        The learned weights of the SVM.

    ``loss_curve_`` : list of float
        List of loss values if show_loss_every > 0.

    ``objective_curve_`` : list of float
       Primal objective after each pass through the dataset.

    ``timestamps_`` : list of int
       Total training time stored before each iteration.

    """
    def __init__(self, model, max_iter=100, C=1.0, verbose=0, momentum=0.,
                 learning_rate='auto', n_jobs=1,
                 show_loss_every=0, decay_exponent=1, decay_t0=10,
                 break_on_no_constraints=True, logger=None, averaging=None):
        SubgradientSSVM.__init__(
            self, model, max_iter, C, verbose=verbose, n_jobs=n_jobs,
            show_loss_every=show_loss_every, decay_exponent=decay_exponent,
            momentum=momentum, learning_rate=learning_rate,
            break_on_no_constraints=break_on_no_constraints, logger=logger,
            decay_t0=decay_t0, averaging=averaging)

    def fit(self, X, Y, H_init=None, warm_start=False, initialize=True):
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

        initialize : boolean, default=True
            Whether to initialize the model for the data.
            Leave this true except if you really know what you are doing.
        """
        if self.verbose > 0:
            print("Training latent subgradient structural SVM")
        if initialize:
            self.model.initialize(X, Y)
        self.grad_old = np.zeros(self.model.size_joint_feature)
        if not warm_start:
            self.w = getattr(self, "w", np.random.normal(
                0, 1, size=self.model.size_joint_feature))
            self.timestamps_ = [time()]
            self.objective_curve_ = []
            if self.learning_rate == "auto":
                self.learning_rate_ = self.C * len(X)
            else:
                self.learning_rate_ = self.learning_rate
        else:
            # hackety hack
            self.timestamps_[0] = time() - self.timestamps_[-1]
        w = self.w.copy()
        n_samples = len(X)
        try:
            # catch ctrl+c to stop training
            for iteration in range(self.max_iter):
                self.timestamps_.append(time() - self.timestamps_[0])
                positive_slacks = 0
                objective = 0.
                #verbose = max(0, self.verbose - 3)

                if self.n_jobs == 1:
                    # online learning
                    for x, y in zip(X, Y):
                        h = self.model.latent(x, y, w)
                        h_hat = self.model.loss_augmented_inference(
                            x, h, w, relaxed=True)
                        delta_joint_feature = (
                            self.model.joint_feature(x, h)
                            - self.model.joint_feature(x, h_hat))
                        slack = (-np.dot(delta_joint_feature, w)
                                 + self.model.loss(h, h_hat))
                        objective += np.maximum(slack, 0)
                        if slack > 0:
                            positive_slacks += 1
                        w = self._solve_subgradient(delta_joint_feature, n_samples, w)
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
                                self.model, x, y, w)
                                for x, y in zip(X_b, Y_b))
                        djoint_feature = np.zeros(self.model.size_joint_feature)
                        for x, y, constraint in zip(X_b, Y_b,
                                                    candidate_constraints):
                            y_hat, delta_joint_feature, slack, loss = constraint
                            objective += slack
                            djoint_feature += delta_joint_feature
                            if slack > 0:
                                positive_slacks += 1
                        djoint_feature /= float(len(X_b))
                        w = self._solve_subgradient(djoint_feature, n_samples, w)

                # some statistics
                objective *= self.C
                objective += np.sum(self.w ** 2) / 2.

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
        self.timestamps_.append(time() - self.timestamps_[0])
        self.objective_curve_.append(self._objective(X, Y))
        if self.logger is not None:
            self.logger(self, 'final')
        if self.verbose:
            if self.objective_curve_:
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

    def _objective(self, X, Y):
        constraints = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose - 1)(delayed(find_constraint_latent)(
                self.model, x, y, self.w)
                for x, y in zip(X, Y))
        slacks = list(zip(*constraints))[2]
        slacks = np.maximum(slacks, 0)

        objective = np.sum(slacks) * self.C + np.sum(self.w ** 2) / 2.
        return objective
