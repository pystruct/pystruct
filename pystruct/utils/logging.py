import cPickle
from time import time

import numpy as np

class SaveLogger(object):
    """Logging class that stores the model periodically.

    Can be used to back up a model during learning.
    Also a prototype to demonstrate the logging interface.

    Parameters
    ----------
    file_name : string
        File in which the model will be stored. If the string contains
        '%d', this will be replaced with the current iteration.

    log_every : int (default=10)
        How often the model should be stored (in iterations).

    verbose : int (default=0)
        Verbosity level.

    """
    def __init__(self, file_name, log_every=10, verbose=0):
        self.file_name = file_name
        self.log_every = log_every
        self.verbose = verbose

    def __repr__(self):
        log_every = getattr(self, "log_every", self.log_every)
        return ('%s(file_name="%s", log_every=%s)'
                % (self.__class__.__name__, self.file_name, log_every))

    def __call__(self, learner, X, Y, iteration=0, force=False):
        """Save learner if iterations is a multiple of log_every or "final".

        Parameters
        ----------
        learner : object
            Learning object to be saved.

        iteration : int (default=0)
            If log_every % iteration == 0,
            the model will be saved.
        force : bool, default=False
            If True, will save independent of iteration.
        """
        if force or not iteration % self.log_every:
            file_name = self.file_name
            if "%d" in file_name:
                file_name = file_name % iteration
            if self.verbose > 0:
                print("saving %s to file %s" % (learner, file_name))
            with open(file_name, "wb") as f:
                if hasattr(learner, 'inference_cache_'):
                    # don't store the large inference cache!
                    learner.inference_cache_, tmp = (None,
                                                     learner.inference_cache_)
                    cPickle.dump(learner, f, -1)
                    learner.inference_cache_ = tmp
                else:
                    cPickle.dump(learner, f, -1)

    def load(self):
        """Load the model stoed in file_name and return it."""
        with open(self.file_name, "rb") as f:
            learner = cPickle.load(f)
        return learner


class AnalysisLogger(SaveLogger):
    """Log everything. """
    def __init__(self, file_name=None, log_every=10, verbose=0, compute_primal=True,
                 compute_loss=True, compute_dual=True, skip_caching=True):
        SaveLogger.__init__(self, file_name=file_name, log_every=log_every,
                            verbose=verbose)
        self.compute_primal = compute_primal
        self.compute_loss = compute_loss
        self.skip_caching = skip_caching
        self.primal_objective_ = []
        self.dual_objective_ = []
        self.timestamps_ = []
        self.loss_ = []
        self.init_time_ = time()
        self.iterations_ = []

    def __repr__(self):
        return ('%s(file_name="%s", log_every=%s)'
                % (self.__class__.__name__, self.file_name, self.log_every))

    def __call__(self, learner, X, Y, iteration=0, force=False):
        """Save learner if iterations is a multiple of log_every or "final".

        Parameters
        ----------
        learner : object
            Learning object to be saved.

        iteration : int (default=0)
            If log_every % iteration == 0,
            the model will be logged.

        force : bool, default=False
            Whether to force logging, such as in the last iteration.
        """
        if self.skip_caching and hasattr(learner, 'cached_constraint_') and not force:
            iteration = iteration - np.sum(learner.cached_constraint_)
            if iteration < len(self.iterations_) * self.log_every:
                # no inference since last log
                return
        if force or not iteration % self.log_every:
            self.iterations_.append(iteration)
            self.timestamps_.append(time() - self.init_time_)
            if self.compute_primal:
                self.primal_objective_.append(learner._objective(X, Y))
            if hasattr(learner, 'dual_objective_curve_'):
                self.dual_objective_.append(learner.dual_objective_curve_[-1])
            if self.compute_loss:
                self.loss_.append(np.sum(learner.model.batch_loss(Y, learner.predict(X))))
        if self.file_name is not None:
            SaveLogger.__call__(self, learner, X, Y, iteration=iteration, force=force)

