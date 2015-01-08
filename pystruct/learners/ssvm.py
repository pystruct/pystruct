
import numpy as np
from sklearn.base import BaseEstimator

from ..utils import inference, inference_map, objective_primal, ParallelMixin


class BaseSSVM(BaseEstimator, ParallelMixin):
    """ABC that implements common functionality."""
    def __init__(self, model, max_iter=100, C=1.0, verbose=0,
                 n_jobs=1, show_loss_every=0, logger=None, 
                 use_threads=False, use_memmapping_pool=1, 
                 memmapping_temp_folder=None):
        self.model = model
        self.max_iter = max_iter
        self.C = C
        self.verbose = verbose
        self.show_loss_every = show_loss_every
        self.n_jobs = n_jobs
        self.logger = logger
        self.use_threads = use_threads
        self.use_memmapping_pool = use_memmapping_pool
        self.memmapping_temp_folder = memmapping_temp_folder 
        self.pool = None


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
        if hasattr(self.model, 'batch_inference'):
            return self.model.batch_inference(X, self.w)
        else:
            prediction = self.parallel(inference_map,
                    ((self.model, x, self.w) for x in X))
            return prediction

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
            losses = self.model.batch_loss(Y, self.predict(X))
        else:
            losses = [self.model.loss(y, y_pred)
                      for y, y_pred in zip(Y, self.predict(X))]
        max_losses = [self.model.max_loss(y) for y in Y]
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

    def _objective(self, X, Y):
        if type(self).__name__ == 'OneSlackSSVM':
            variant = 'one_slack'
        else:
            variant = 'n_slack'
        if self.pool is None:
            self._spawn_pool()
        return objective_primal(self.model, 
                self.w, X, Y, self.C, 
                variant=variant, parallel=self.parallel)

