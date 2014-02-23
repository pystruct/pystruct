import numpy as np


class StructuredModel(object):
    """Interface definition for Structured Learners.

    This class defines what is necessary to use the structured svm.
    You have to implement at least joint_feature and inference.
    """
    def __repr__(self):
        return ("%s, size_joint_feature: %d"
                % (type(self).__name__, self.size_joint_feature))

    def __init__(self):
        """Initialize the model.
        Needs to set self.size_joint_feature, the dimensionalty of the joint features for
        an instance with labeling (x, y).
        """
        self.size_joint_feature = None

    def _check_size_w(self, w):
        if w.shape != (self.size_joint_feature,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             (self.size_joint_feature, w.shape))

    def initialize(self, X, Y):
        # set any data-specific parameters in the model
        pass

    def joint_feature(self, x, y):
        raise NotImplementedError()

    def batch_joint_feature(self, X, Y, Y_true=None):
        joint_feature_ = np.zeros(self.size_joint_feature)
        if getattr(self, 'rescale_C', False):
            for x, y, y_true in zip(X, Y, Y_true):
                joint_feature_ += self.joint_feature(x, y, y_true)
        else:
            for x, y in zip(X, Y):
                joint_feature_ += self.joint_feature(x, y)
        return joint_feature_

    def _loss_augmented_djoint_feature(self, x, y, y_hat, w):
        # debugging only!
        x_loss_augmented = self.loss_augment(x, y, w)
        return (self.joint_feature(x_loss_augmented, y)
                - self.joint_feature(x_loss_augmented, y_hat))

    def inference(self, x, w, relaxed=None):
        raise NotImplementedError()

    def batch_inference(self, X, w, relaxed=None):
        # default implementation of batch inference
        return [self.inference(x, w, relaxed=relaxed)
                for x in X]

    def loss(self, y, y_hat):
        # hamming loss:
        if isinstance(y_hat, tuple):
            return self.continuous_loss(y, y_hat[0])
        if hasattr(self, 'class_weight'):
            return np.sum(self.class_weight[y] * (y != y_hat))
        return np.sum(y != y_hat)

    def batch_loss(self, Y, Y_hat):
        # default implementation of batch loss
        return [self.loss(y, y_hat) for y, y_hat in zip(Y, Y_hat)]

    def max_loss(self, y):
        # maximum possible los on y for macro averages
        if hasattr(self, 'class_weight'):
            return np.sum(self.class_weight[y])
        return y.size

    def continuous_loss(self, y, y_hat):
        # continuous version of the loss
        # y is the result of linear programming
        if y.ndim == 2:
            raise ValueError("FIXME!")
        gx = np.indices(y.shape)

        # all entries minus correct ones
        result = 1 - y_hat[gx, y]
        if hasattr(self, 'class_weight'):
            return np.sum(self.class_weight[y] * result)
        return np.sum(result)

    def loss_augmented_inference(self, x, y, w, relaxed=None):
        print("FALLBACK no loss augmented inference found")
        return self.inference(x, w)

    def batch_loss_augmented_inference(self, X, Y, w, relaxed=None):
        # default implementation of batch loss augmented inference
        return [self.loss_augmented_inference(x, y, w, relaxed=relaxed)
                for x, y in zip(X, Y)]

    def _set_class_weight(self):
        if not hasattr(self, 'size_joint_feature'):
            # we are not initialized yet
            return

        if hasattr(self, 'n_labels'):
            n_things = self.n_labels
        else:
            n_things = self.n_states

        if self.class_weight is not None:

            if len(self.class_weight) != n_things:
                raise ValueError("class_weight must have length n_states or"
                                 " be None")
            self.class_weight = np.array(self.class_weight)
            self.uniform_class_weight = False
        else:
            self.class_weight = np.ones(n_things)
            self.uniform_class_weight = True
