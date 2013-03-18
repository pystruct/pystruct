
import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import BaseEstimator

from ..utils import inference


class BaseSSVM(BaseEstimator):
    """ABC that implements common functionality."""
    def __init__(self, problem, max_iter=100, C=1.0, verbose=0,
                 n_jobs=1, show_loss_every=0, logger=None):
        self.problem = problem
        self.max_iter = max_iter
        self.C = C
        self.verbose = verbose
        self.show_loss_every = show_loss_every
        self.n_jobs = n_jobs
        self.logger = logger

    def predict(self, X):
        """Predict output on examples in X.
        Parameters
        ----------
        X : iterable
            Traing instances. Contains the structured input objects.

        Returns
        -------
        Y_pred : list
            List of inference results for X using the learned parameters.
        """
        verbose = max(0, self.verbose - 3)
        if self.n_jobs != 1:
            prediction = Parallel(n_jobs=self.n_jobs, verbose=verbose)(
                delayed(inference)(self.problem, x, self.w) for x in X)
            return prediction
        else:
            if hasattr(self.problem, 'batch_inference'):
                return self.problem.batch_inference(X, self.w)
            return [self.problem.inference(x, self.w) for x in X]

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
        if hasattr(self.problem, 'batch_loss'):
            losses = self.problem.batch_loss(Y, self.predict(X))
        else:
            losses = [self.problem.loss(y, y_pred)
                      for y, y_pred in zip(Y, self.predict(X))]
        max_losses = [self.problem.max_loss(y) for y in Y]
        return 1. - np.sum(losses) / float(np.sum(max_losses))

    def _compute_training_loss(self, X, Y, iteration):
        # optionally compute training loss for output / training curve
        if (self.show_loss_every != 0
                and not iteration % self.show_loss_every):
            if not hasattr(self, 'loss_curve_'):
                self.loss_curve_ = []
            display_loss = 1 - self.score(X, Y)
            if self.verbose > 0:
                print("current loss: %f" % (display_loss))
            self.loss_curve_.append(display_loss)
