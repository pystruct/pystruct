import numpy as np
from sklearn.externals.joblib import Parallel, delayed

from .ssvm import BaseSSVM


def inference(model, x, w):
    return model.inference(x, w)


class StructuredPerceptron(BaseSSVM):
    """Structured Perceptron training.

    Implements a simple structured perceptron with optional averaging.
    The structured perceptron approximately minimizes the zero-one loss,
    therefore the learning does not take ``model.loss`` into account. It is
    just shown to illustrate the learning progress.

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

    decay_t0 : float, default=10
        Offset for decaying learning rate. Effective learning rate is
        ``(t0 + t)** decay_exponent``. Zero means no decay.

    average : bool or int, default=False
        Whether to average over all weight vectors obtained during training
        or simply keeping the last one.
        ``average=False`` does not perform any averaging.
        ``average=True`` averages over all epochs.
        ``average=k`` with ``k >= 0`` waits ``k`` epochs before averaging.
        ``average=k`` with ``k < 0`` averages over the last ``k`` epochs.  So
        far ``k = -1`` is the only negative value supported.

    logger : logger object.

    Attributes
    ----------
    w : nd-array, shape=(model.size_joint_feature,)
        The learned weights of the SVM.

   ``loss_curve_`` : list of float
        List of loss values after each pass thorugh the dataset.

    References
    ----------
    Michael Collins. Discriminative training methods for hidden Markov models:
        theory and experiments with perceptron algorithms. In Proc. EMNLP 2002
        http://www.aclweb.org/anthology-new/W/W02/W02-1001.pdf

    """
    def __init__(self, model, max_iter=100, verbose=0, batch=False,
                 decay_exponent=0, decay_t0=10, average=False, n_jobs=1,
                 logger=None):
        BaseSSVM.__init__(self, model, max_iter=max_iter, verbose=verbose,
                          n_jobs=n_jobs, logger=logger)
        self.batch = batch
        self.decay_exponent = decay_exponent
        self.decay_t0 = decay_t0
        self.average = average

    def fit(self, X, Y, initialize=True):
        """Learn parameters using structured perceptron.

        Parameters
        ----------
        X : iterable
            Traing instances. Contains the structured input objects.
            No requirement on the particular form of entries of X is made.

        Y : iterable
            Training labels. Contains the strctured labels for inputs in X.
            Needs to have the same length as X.

        initialize : boolean, default=True
            Whether to initialize the model for the data.
            Leave this true except if you really know what you are doing.
        """
        if initialize:
            self.model.initialize(X, Y)
        size_joint_feature = self.model.size_joint_feature
        self.w = np.zeros(size_joint_feature)
        if self.average is not False:
            if self.average is True:
                self.average = 0
            elif self.average < -1:
                raise NotImplemented("The only negative value for averaging "
                                     "implemented at the moment is `-1`. Try "
                                     "`max_iter - k` but be aware of the "
                                     "possibility of early stopping.")
            w_bar = np.zeros(size_joint_feature)
            n_obs = 0
        self.loss_curve_ = []
        max_losses = np.sum([self.model.max_loss(y) for y in Y])
        try:
            for iteration in range(self.max_iter):
                if self.average == -1:
                    # By resetting at every iteration we effectively get
                    # averaging over the last one.
                    n_obs = 0
                    w_bar.fill(0)
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
                            self.w += effective_lr * (self.model.joint_feature(x, y) -
                                                      self.model.joint_feature(x, y_hat))
                    if self.average is not False and iteration >= self.average:
                        n_obs += 1
                        w_bar = ((1 - 1. / n_obs) * w_bar +
                                 (1. / n_obs) * self.w)
                else:
                    # standard online update
                    for x, y in zip(X, Y):
                        y_hat = self.model.inference(x, self.w)
                        current_loss = self.model.loss(y, y_hat)
                        losses += current_loss
                        if current_loss:
                            self.w += effective_lr * (self.model.joint_feature(x, y) -
                                                      self.model.joint_feature(x, y_hat))
                        if (self.average is not False and
                                iteration >= self.average):
                            n_obs += 1
                            w_bar = ((1 - 1. / n_obs) * w_bar +
                                     (1. / n_obs) * self.w)
                self.loss_curve_.append(float(losses) / max_losses)
                if self.verbose:
                    print("avg loss: %f w: %s" % (self.loss_curve_[-1],
                                                  str(self.w)))
                    print("effective learning rate: %f" % effective_lr)
                if self.loss_curve_[-1] == 0:
                    if self.verbose:
                        print("Loss zero. Stopping.")
                    break

        except KeyboardInterrupt:
            pass
        finally:
            if self.average is not False:
                self.w = w_bar
        return self
