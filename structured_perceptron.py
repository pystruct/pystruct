import numpy as np
import matplotlib.pyplot as plt


class StructuredPerceptron(object):
    def __init__(self, problem, max_iter=100):
        self.max_iter = max_iter
        self.problem = problem

    def fit(self, X, Y):
        n_samples = len(X)
        size_psi = self.problem.size_psi
        w = np.zeros(size_psi)
        try:
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
                print("avg loss: %f w: %s" %
                        (float(losses) / n_samples, str(w)))
        except KeyboardInterrupt:
            pass
        self.w = w

    def predict(self, X):
        prediction = []
        for x in X:
            prediction.append(self.problem.inference(x, self.w))
        return prediction


class LatentStructuredPerceptron(StructuredPerceptron):
    def fit(self, X, Y):
        n_samples = len(X)
        size_psi = self.problem.size_psi
        w = np.zeros(size_psi)
        try:
            for iteration in xrange(self.max_iter):
                alpha = 0.01 / (1 + iteration)
                losses = 0
                print("iteration %d" % iteration)
                i = 0
                for x, y in zip(X, Y):
                    print("example %03d" % i)
                    h = self.problem.latent(x, y, w)
                    plt.matshow(h.reshape(33, 39))
                    plt.savefig("h_%03d_%03d.png" % (iteration, i))
                    plt.close()
                    h_hat, y_hat = self.problem.inference(x, w)
                    plt.matshow(h_hat.reshape(33, 39))
                    plt.savefig("h_hat_%03d_%03d.png" % (iteration, i))
                    plt.close()
                    current_loss = self.problem.loss(y, y_hat)
                    losses += current_loss
                    if current_loss:
                        w += alpha * (self.problem.psi(x, h, y)
                                - self.problem.psi(x, h_hat, y_hat))
                    i += 1
                print("avg loss: %f w: %s" %
                        (float(losses) / n_samples, str(w)))
        except KeyboardInterrupt:
            pass
        self.w = w

    def predict(self, X):
        prediction = []
        for x in X:
            prediction.append(self.problem.inference(x, self.w))
        return prediction
