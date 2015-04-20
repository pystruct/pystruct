from time import time
import numpy as np

from sklearn.externals.joblib import Parallel, delayed, cpu_count
from sklearn.utils import gen_even_slices, shuffle

from .ssvm import BaseSSVM
from ..utils import find_constraint


class SubgradientSSVM(BaseSSVM):
    """Structured SVM solver using subgradient descent.

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
        Regularization parameter.

    verbose : int, default=0
        Verbosity.

    learning_rate : float or 'auto', default='auto'
        Learning rate used in subgradient descent. If 'auto', the pegasos
        schedule is used, which starts with ``learning_rate = n_samples * C``.

    momentum : float, default=0.0
        Momentum used in subgradient descent.

    n_jobs : int, default=1
        Number of parallel jobs for inference. -1 means as many as cpus.

    batch_size : int, default=None
        Ignored if n_jobs > 1. If n_jobs=1, inference will be done in mini
        batches of size batch_size. If n_jobs=-1, batch learning will be
        performed, that is the whole dataset will be used to compute each
        subgradient.

    show_loss_every : int, default=0
        Controlls how often the hamming loss is computed (for monitoring
        purposes). Zero means never, otherwise it will be computed very
        show_loss_every'th epoch.

    decay_exponent : float, default=1
        Exponent for decaying learning rate. Effective learning rate is
        ``learning_rate / (decay_t0 + t)** decay_exponent``. Zero means no
        decay.

    decay_t0 : float, default=10
        Offset for decaying learning rate. Effective learning rate is
        ``learning_rate / (decay_t0 + t)** decay_exponent``.

    break_on_no_constraints : bool, default=True
        Break when there are no new constraints found.

    logger : logger object.

    averaging : string, default=None
        Whether and how to average weights. Possible options are 'linear',
        'squared' and None.
        The string reflects the weighting of the averaging:

            - ``linear: w_avg ~ w_1 + 2 * w_2 + ... + t * w_t``

            - ``squared: w_avg ~ w_1 + 4 * w_2 + ... + t**2 * w_t``

        Uniform averaging is not implemented as it is worse than linear
        weighted averaging or no averaging.

    shuffle : bool, default=False
        Whether to shuffle the dataset in each iteration.

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

    References
    ----------
    * Nathan Ratliff, J. Andrew Bagnell and Martin Zinkevich:
        (Online) Subgradient Methods for Structured Prediction, AISTATS 2007

    * Shalev-Shwartz, Shai and Singer, Yoram and Srebro, Nathan and Cotter,
        Andrew: Pegasos: Primal estimated sub-gradient solver for svm,
        Mathematical Programming 2011
    """
    def __init__(self, model, max_iter=100, C=1.0, verbose=0, momentum=0.0,
                 learning_rate='auto', n_jobs=1,
                 show_loss_every=0, decay_exponent=1,
                 break_on_no_constraints=True, logger=None, batch_size=None,
                 decay_t0=10, averaging=None, shuffle=False):
        BaseSSVM.__init__(self, model, max_iter, C, verbose=verbose,
                          n_jobs=n_jobs, show_loss_every=show_loss_every,
                          logger=logger)
        self.averaging = averaging
        self.break_on_no_constraints = break_on_no_constraints
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.t = 0
        self.decay_exponent = decay_exponent
        self.decay_t0 = decay_t0
        self.batch_size = batch_size
        self.shuffle = shuffle

    def _solve_subgradient(self, djoint_feature, n_samples, w):
        """Do a single subgradient step."""
        grad = (djoint_feature - w / (self.C * n_samples))

        self.grad_old = ((1 - self.momentum) * grad
                         + self.momentum * self.grad_old)
        if self.decay_exponent == 0:
            effective_lr = self.learning_rate_
        else:
            effective_lr = (self.learning_rate_
                            / (self.t + self.decay_t0)
                            ** self.decay_exponent)
        w += effective_lr * self.grad_old

        if self.averaging == 'linear':
            rho = 2. / (self.t + 2.)
            self.w = (1. - rho) * self.w + rho * w
        elif self.averaging == 'squared':
            rho = 6. * (self.t + 1) / ((self.t + 2) * (2 * self.t + 3))
            self.w = (1. - rho) * self.w + rho * w
        else:
            self.w = w
        self.t += 1.
        return w

    def fit(self, X, Y, constraints=None, warm_start=False, initialize=True):
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
        if initialize:
            self.model.initialize(X, Y)
        if self.verbose:
            print("Training primal subgradient structural SVM")
        self.grad_old = np.zeros(self.model.size_joint_feature)
        self.w = getattr(self, "w", np.zeros(self.model.size_joint_feature))
        w = self.w.copy()
        if not warm_start:
            self.objective_curve_ = []
            self.timestamps_ = [time()]
            if self.learning_rate == "auto":
                self.learning_rate_ = self.C * len(X)
            else:
                self.learning_rate_ = self.learning_rate
        else:
            self.timestamps_ = (np.array(self.timestamps_) - time()).tolist()
        try:
            # catch ctrl+c to stop training
            for iteration in range(self.max_iter):
                if self.shuffle:
                    X, Y = shuffle(X, Y)
                if self.n_jobs == 1:
                    objective, positive_slacks, w = self._sequential_learning(X, Y, w)
                else:
                    objective, positive_slacks, w = self._parallel_learning(X, Y, w)

                # some statistics
                objective = objective * self.C + np.sum(w ** 2) / 2.

                if positive_slacks == 0:
                    if self.verbose:
                        print("No additional constraints")
                    if self.break_on_no_constraints:
                        break
                if self.verbose > 0:
                    print(self)
                    print("iteration %d" % iteration)
                    print("positive slacks: %d,"
                          "objective: %f" %
                          (positive_slacks, objective))
                self.timestamps_.append(time() - self.timestamps_[0])
                self.objective_curve_.append(self._objective(X, Y))

                if self.verbose > 2:
                    print(self.w)

                self._compute_training_loss(X, Y, iteration)
                if self.logger is not None:
                    self.logger(self, iteration)

        except KeyboardInterrupt:
            pass

        if self.verbose:
            print("Computing final objective")

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

    def _parallel_learning(self, X, Y, w):
        n_samples = len(X)
        objective, positive_slacks = 0, 0
        verbose = max(0, self.verbose - 3)
        if self.batch_size is not None:
            raise ValueError("If n_jobs != 1, batch_size needs to"
                             "be None")
        # generate batches of size n_jobs
        # to speed up inference
        if self.n_jobs == -1:
            n_jobs = cpu_count()
        else:
            n_jobs = self.n_jobs

        n_batches = int(np.ceil(float(len(X)) / n_jobs))
        slices = gen_even_slices(n_samples, n_batches)
        for batch in slices:
            X_b = X[batch]
            Y_b = Y[batch]
            candidate_constraints = Parallel(
                n_jobs=self.n_jobs,
                verbose=verbose)(delayed(find_constraint)(
                    self.model, x, y, w)
                    for x, y in zip(X_b, Y_b))
            djoint_feature = np.zeros(self.model.size_joint_feature)
            for x, y, constraint in zip(X_b, Y_b,
                                        candidate_constraints):
                y_hat, delta_joint_feature, slack, loss = constraint
                if slack > 0:
                    objective += slack
                    djoint_feature += delta_joint_feature
                    positive_slacks += 1
            w = self._solve_subgradient(djoint_feature, n_samples, w)
        return objective, positive_slacks, w

    def _sequential_learning(self, X, Y, w):
        n_samples = len(X)
        objective, positive_slacks = 0, 0
        if self.batch_size in [None, 1]:
            # online learning
            for x, y in zip(X, Y):
                y_hat, delta_joint_feature, slack, loss = \
                    find_constraint(self.model, x, y, w)
                objective += slack
                if slack > 0:
                    positive_slacks += 1
                self._solve_subgradient(delta_joint_feature, n_samples, w)
        else:
            # mini batch learning
            if self.batch_size == -1:
                slices = [slice(0, len(X))]
            else:
                n_batches = int(np.ceil(float(len(X)) / self.batch_size))
                slices = gen_even_slices(n_samples, n_batches)
            for batch in slices:
                X_b = X[batch]
                Y_b = Y[batch]
                Y_hat = self.model.batch_loss_augmented_inference(
                    X_b, Y_b, w, relaxed=True)
                delta_joint_feature = (self.model.batch_joint_feature(X_b, Y_b)
                                       - self.model.batch_joint_feature(X_b, Y_hat))
                loss = np.sum(self.model.batch_loss(Y_b, Y_hat))

                violation = np.maximum(0, loss - np.dot(w, delta_joint_feature))
                objective += violation
                positive_slacks += self.batch_size
                self._solve_subgradient(delta_joint_feature / len(X_b), n_samples, w)
        return objective, positive_slacks, w
