
import numpy as np

from sklearn.externals.joblib import Parallel, delayed, cpu_count
from sklearn.utils import gen_even_slices

from .subgradient_ssvm import SubgradientStructuredSVM
from ..utils import find_constraint_latent


class LatentSubgradientSSVM(SubgradientStructuredSVM):
    """Latent Variable Structured SVM solver using subgradient descent.

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

    show_loss_every : int, default=0
        Controlls how often the hamming loss is computed (for monitoring
        purposes). Zero means never, otherwise it will be computed very
        show_loss_every'th epoch.

    decay_exponent : float, default=0
        Exponent for decaying learning rate. Effective learning rate is
        ``learning_rate / t ** decay_exponent``. Zero means no decay.
        Ignored if adagrad=True.


    Attributes
    ----------
    w : nd-array, shape=(problem.psi,)
        The learned weights of the SVM.

   ``loss_curve_`` : list of float
        List of loss values if show_loss_every > 0.

   ``objective_curve_`` : list of float
       Primal objective after each pass through the dataset.

    """
    def __init__(self, problem, max_iter=100, C=1.0, verbose=0, momentum=0.9,
                 learning_rate=0.001, adagrad=False, n_jobs=1,
                 show_loss_every=0, decay_exponent=0):
        SubgradientStructuredSVM.__init__(
            self, problem, max_iter, C, verbose=verbose, n_jobs=n_jobs,
            show_loss_every=show_loss_every, decay_exponent=decay_exponent,
            momentum=momentum, learning_rate=learning_rate, adagrad=adagrad)

    def fit(self, X, Y, H_init=None):
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
        """
        print("Training latent subgradient structural SVM")
        #w = getattr(self, "w", np.zeros(self.problem.size_psi))
        w = getattr(self, "w", np.random.normal(0, .001,
                                                size=self.problem.size_psi))
        #constraints = []
        objective_curve = []
        n_samples = len(X)
        try:
            # catch ctrl+c to stop training
            for iteration in xrange(self.max_iter):
                positive_slacks = 0
                objective = 0.
                #verbose = max(0, self.verbose - 3)

                if self.n_jobs == 1:
                    # online learning
                    for x, y in zip(X, Y):
                        h = self.problem.latent(x, y, w)
                        h_hat = self.problem.loss_augmented_inference(
                            x, h, w, relaxed=True)
                        delta_psi = (self.problem.psi(x, h)
                                     - self.problem.psi(x, h_hat))
                        slack = (-np.dot(delta_psi, w)
                                 + self.problem.loss(h, h_hat))
                        objective += np.maximum(slack, 0)
                        if slack > 0:
                            positive_slacks += 1
                        w = self._solve_subgradient(w, delta_psi,
                                                    n_samples)
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
                                self.problem, x, y, w)
                                for x, y in zip(X_b, Y_b))
                        dpsi = np.zeros(self.problem.size_psi)
                        for x, y, constraint in zip(X_b, Y_b,
                                                    candidate_constraints):
                            y_hat, delta_psi, slack, loss = constraint
                            objective += slack
                            dpsi += delta_psi
                            if slack > 0:
                                positive_slacks += 1
                        dpsi /= float(len(X_b))
                        w = self._solve_subgradient(w, dpsi, n_samples)

                # some statistics
                objective += np.sum(w ** 2) / self.C / 2.
                objective /= float(n_samples)

                if positive_slacks == 0:
                    print("No additional constraints")
                    from IPython.core.debugger import Tracer
                    Tracer()()
                    break
                if self.verbose > 0:
                    print(self)
                    print("iteration %d" % iteration)
                    print("positive slacks: %d,"
                          "objective: %f" %
                          (positive_slacks, objective))
                objective_curve.append(objective)

                if self.verbose > 2:
                    print(w)

                self._compute_training_loss(X, Y, w, iteration)

        except KeyboardInterrupt:
            pass
        self.w = w
        self.objective_curve_ = objective_curve
        print("final objective: %f" % objective_curve[-1])
        print("calls to inference: %d" % self.problem.inference_calls)
        return self

    def predict(self, X):
        prediction = SubgradientStructuredSVM.predict(self, X)
        return [self.problem.label_from_latent(h) for h in prediction]

    def predict_latent(self, X):
        return SubgradientStructuredSVM.predict(self, X)

    def score(self, X, Y):
        """Compute score as 1 - loss over whole data set.

        Returns the average accuracy (in terms of problem.loss)
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
        if hasattr(self.problem, 'batch_batch_loss'):
            losses = self.problem.base_batch_loss(Y, self.predict(X))
        else:
            losses = [self.problem.base_loss(y, y_pred)
                      for y, y_pred in zip(Y, self.predict(X))]
        max_losses = [self.problem.max_loss(y) for y in Y]
        return 1. - np.sum(losses) / float(np.sum(max_losses))
