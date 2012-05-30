import numpy as np


class StructuredPerceptron(object):
    def __init__(self, problem, max_iter=100):
        self.max_iter = max_iter
        self.problem = problem

    def fit(self, X, Y):
        size_psi = self.problem.size_psi
        w = np.zeros(size_psi)
        for iteration in xrange(self.max_iter):
            alpha = 1. / (1 + iteration)
            losses = 0
            print("iteration %d" % iteration)
            for x, y in zip(X, Y):
                y_hat = self.problem.inference(x, w)
                current_loss = self.problem.loss(y, y_hat)
                losses += current_loss
                if current_loss:
                    w += alpha * (self.problem.psi(x, y)
                            - self.problem.psi(x, y_hat))
            print("loss: %f w: %s" % (losses, str(w)))
        self.w = w

    def predict(self, X):
        prediction = []
        for x in X:
            prediction.append(self.problem.inference(x, self.w))
        return prediction
