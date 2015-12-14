try:
    import cPickle as pickle
except ImportError:
    import pickle


class SaveLogger(object):
    """Logging class that stores the model periodically.

    Can be used to back up a model during learning.
    Also a prototype to demonstrate the logging interface.

    Parameters
    ----------
    file_name : string
        File in which the model will be stored. If the string contains
        '%d', this will be replaced with the current iteration.

    save_every : int (default=10)
        How often the model should be stored (in iterations).

    verbose : int (default=0)
        Verbosity level.

    """
    def __init__(self, file_name, save_every=10, verbose=0):
        self.file_name = file_name
        self.save_every = save_every
        self.verbose = verbose

    def __repr__(self):
        return ('%s(file_name="%s", save_every=%s)'
                % (self.__class__.__name__, self.file_name, self.save_every))

    def __call__(self, learner, iteration=0):
        """Save learner if iterations is a multiple of save_every or "final".

        Parameters
        ----------
        learner : object
            Learning object to be saved.

        iteration : int or 'final' (default=0)
            If 'final' or save_every % iteration == 0,
            the model will be saved.
        """
        if iteration == 'final' or not iteration % self.save_every:
            file_name = self.file_name
            if "%d" in file_name:
                file_name = file_name % iteration
            if self.verbose > 0:
                print("saving %s to file %s" % (learner, file_name))
            self.save(learner, file_name)

    def save(self, learner, file_name):
        """Save the model to location specified in file_name."""
        with open(file_name, "wb") as f:
            if hasattr(learner, 'inference_cache_'):
                # don't store the large inference cache!
                learner.inference_cache_, tmp = (None,
                                                 learner.inference_cache_)
                pickle.dump(learner, f, -1)
                learner.inference_cache_ = tmp
            else:
                pickle.dump(learner, f, -1)

    def load(self):
        """Load the model stoed in file_name and return it."""
        with open(self.file_name, "rb") as f:
            learner = pickle.load(f)
        return learner
