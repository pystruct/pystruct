import numpy as np
from sklearn.externals.joblib import Parallel, delayed

from .ssvm import BaseSSVM


def inference(model, x, w):
    return model.inference(x, w)


class StructuredPerceptron(BaseSSVM):
    """Structured Perceptron training.

    Implements a simple structured perceptron.
    The structured perceptron approximately minimizes the zero-one loss.
    Therefore the learning does not take model.loss into account.
    It is just shown to illustrate the learning progress.
    As the perceptron learning is not margin-based, the model does not
    need to provide loss_augmented_inference.

    Parameters
    ----------
    model : StructuredModel
        Object containing model structure. Has to implement
        `loss`, `inference`.

    max_iter : int (default=100)
        Maximum number of passes over dataset to find constraints and update
        parameters.

    verbose : int (default=0)
        Verbosity

    batch : bool (default=False)
        Whether to do batch learning or online learning.

    decay_exponent : float, default=0
        Exponent for decaying learning rate. Effective learning rate is
        ``(t0 + t)** decay_exponent``. Zero means no decay.
        Ignored if adagrad=True.

    decay_t0 : float, default=10
        Offset for decaying learning rate. Effective learning rate is
        ``(t0 + t)** decay_exponent``. Zero means no decay.
        Ignored if adagrad=True.

    logger : logger object.

    Attributes
    ----------
    w : nd-array, shape=(model.psi,)
        The learned weights of the SVM.

   ``loss_curve_`` : list of float
        List of loss values after each pass thorugh the dataset.
    """
    def __init__(self, model, max_iter=100, verbose=0, batch=False,
                 decay_exponent=0, decay_t0=10, n_jobs=1, logger=None):
        BaseSSVM.__init__(self, model, max_iter=max_iter, verbose=verbose,
                          n_jobs=n_jobs, logger=logger)
        self.batch = batch
        self.decay_exponent = decay_exponent
        self.decay_t0 = decay_t0

    def fit(self, X, Y):
        """Learn parameters using structured perceptron.

        Parameters
        ----------
        X : iterable
            Traing instances. Contains the structured input objects.
            No requirement on the particular form of entries of X is made.

        Y : iterable
            Training labels. Contains the strctured labels for inputs in X.
            Needs to have the same length as X.
        """

        n_samples = len(X)
        size_psi = self.model.size_psi
        self.w = np.zeros(size_psi)
        self.loss_curve_ = []
        try:
            for iteration in xrange(self.max_iter):
                effective_lr = ((iteration + self.decay_t0) **
                                self.decay_exponent)
                losses = 0
                if self.verbose:
                    print("iteration %d" % iteration)
                if self.batch:
                    Y_hat = (Parallel(n_jobs=self.n_jobs)(
                        delayed(inference)(self.model, x, self.w) for x, y in
                        zip(X, Y)))
                    for x, y, y_hat in zip(X, Y, Y_hat):
                        current_loss = self.model.loss(y, y_hat)
                        losses += current_loss
                        if current_loss:
                            self.w += effective_lr * (self.model.psi(x, y) -
                                                      self.model.psi(x, y_hat))
                else:
                    # standard online update
                    for x, y in zip(X, Y):
                        y_hat = self.model.inference(x, self.w)
                        current_loss = self.model.loss(y, y_hat)
                        losses += current_loss
                        if current_loss:
                            self.w += effective_lr * (self.model.psi(x, y) -
                                                      self.model.psi(x, y_hat))
                self.loss_curve_.append(float(losses) / n_samples)
                if self.verbose:
                    print("avg loss: %f w: %s" % (self.loss_curve_[-1],
                                                  str(self.w)))
                    print("effective learning rate: %f" % effective_lr)
                if self.loss_curve_[-1] == 0:
                    print("Loss zero. Stopping.")
                    break
        except KeyboardInterrupt:
            pass
        return self
