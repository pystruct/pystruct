import numpy as np
from sklearn.externals.joblib import Parallel, delayed

from ..utils import inference


class BaseSSVM(object):
    """ABC that implements common functionality."""
    def __init__(self, problem, max_iter=100, C=1.0, verbose=0,
                 n_jobs=1, show_loss='augmented'):
        self.problem = problem
        self.max_iter = max_iter
        self.C = C
        self.verbose = verbose
        self.show_loss = show_loss
        self.n_jobs = n_jobs

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
        prediction = Parallel(n_jobs=self.n_jobs, verbose=verbose)(
            delayed(inference)(self.problem, x, self.w) for x in X)
        return prediction

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
        losses = [self.problem.loss(y, y_pred)
                  for y, y_pred in zip(Y, self.predict(X))]
        max_losses = [self.problem.max_loss(y) for y in Y]
        return 1. - np.sum(losses) / float(np.sum(max_losses))

    def _get_loss(self, x, y, w, augmented_loss):
        if self.show_loss == 'augmented':
            return augmented_loss
        elif self.show_loss == 'true':
            return self.problem.loss(y, self.problem.inference(x, w))
        else:
            raise ValueError("show_loss should be 'augmented' or"
                             " 'true', got %s" % self.show_loss)
