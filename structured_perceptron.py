import numpy as np
import matplotlib.pyplot as plt

from IPython.core.debugger import Tracer
tracer = Tracer()


class StructuredPerceptron(object):
    def __init__(self, problem, max_iter=100, verbose=0, plot=False):
        self.max_iter = max_iter
        self.problem = problem
        self.verbose = verbose
        self.plot = plot

    def fit(self, X, Y):
        n_samples = len(X)
        size_psi = self.problem.size_psi
        w = np.zeros(size_psi)
        loss_curve = []
        try:
            for iteration in xrange(self.max_iter):
                alpha = 1. / (1 + iteration)
                losses = 0
                if self.verbose:
                    print("iteration %d" % iteration)
                for x, y in zip(X, Y):
                    y_hat = self.problem.inference(x, w)
                    current_loss = self.problem.loss(y, y_hat)
                    losses += current_loss
                    if current_loss:
                        w += alpha * (self.problem.psi(x, y) -
                                      self.problem.psi(x, y_hat))
                loss_curve.append(float(losses) / n_samples)
                if self.verbose:
                    print("avg loss: %f w: %s" % (loss_curve[-1], str(w)))
                    print("alpha: %f" % alpha)
        except KeyboardInterrupt:
            pass
        if self.plot:
            plt.plot(loss_curve)
            plt.show()
        self.loss_curve_ = loss_curve
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
        #try:
        for iteration in xrange(self.max_iter):
            alpha = 1. / (1 + iteration)
            losses = 0
            print("iteration %d" % iteration)
            i = 0
            for x, y in zip(X, Y):
                print("example %03d" % i)
                h = self.problem.latent(x, y, w)
                h_hat, y_hat = self.problem.inference(x, w)
                plt.matshow(h_hat.reshape(18, 18))
                if not iteration % 10 and i < 5:
                    plt.matshow(h.reshape(18, 18))
                    plt.colorbar()
                    plt.savefig("figures/h_%03d_%03d.png" % (iteration, i))
                    plt.close()
                    plt.colorbar()
                    plt.savefig("figures/h_hat_%03d_%03d.png" % (iteration, i))
                    plt.close()
                current_loss = self.problem.loss(y, y_hat)
                losses += current_loss
                if current_loss:
                    w += alpha * (self.problem.psi(x, h, y) -
                                  self.problem.psi(x, h_hat, y_hat))
                i += 1
            print("avg loss: %f w: %s" % (float(losses) / n_samples, str(w)))
        #except KeyboardInterrupt:
            #pass
        self.w = w

    def predict(self, X):
        prediction = []
        for x in X:
            prediction.append(self.problem.inference(x, self.w)[0])
        return prediction
