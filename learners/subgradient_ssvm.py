import numpy as np
import matplotlib.pyplot as plt

from .cutting_plane_ssvm import StructuredSVM
from .tools import find_constraint


class SubgradientStructuredSVM(StructuredSVM):
    """Margin rescaled with l1 slack penalty."""
    def __init__(self, problem, max_iter=100, C=1.0, verbose=0, momentum=0.9,
                 learningrate=0.001, plot=False, adagrad=False):
        super(SubgradientStructuredSVM, self).__init__(problem, max_iter, C,
                                                       verbose=verbose)
        self.momentum = momentum
        self.learningrate = learningrate
        self.t = 0
        self.plot = plot
        self.adagrad = adagrad
        self.grad_old = np.zeros(self.problem.size_psi)

    def _solve_subgradient(self, psis):
        if hasattr(self, 'w'):
            w = self.w
        else:
            w = np.ones(self.problem.size_psi) * 1e-10
        psi_matrix = np.vstack(psis).mean(axis=0)
        #w += 1. / self.t * (psi_matrix - w / self.C / 2)
        #grad = (self.learningrate / (self.t + 1.) ** 2
                #* (psi_matrix - w / self.C / 2))
        grad = (psi_matrix - w / self.C)

        if self.adagrad:
            self.grad_old += grad ** 2
            w += self.learningrate * grad / (1. + np.sqrt(self.grad_old))
            print("grad old %f" % np.mean(self.grad_old))
            print("effective lr %f" % (self.learningrate /
                                       np.mean(1. + np.sqrt(self.grad_old))))
        else:
            grad_old = ((1 - self.momentum) * grad
                        + self.momentum * self.grad_old)
            #w += self.learningrate / (self.t + 1) * grad_old
            w += self.learningrate * grad_old

        self.w = w
        self.t += 1.
        return w

    def fit(self, X, Y):
        print("Training primal subgradient structural SVM")
        # we initialize with a small value so that loss-augmented inference
        # can give us something meaningful in the first iteration
        w = 1e-5 * np.ones(self.problem.size_psi)
        #constraints = []
        all_psis = []
        losses = []
        loss_curve = []
        objective_curve = []
        for iteration in xrange(self.max_iter):
            psis = []
            positive_slacks = 0
            current_loss = 0.
            objective = 0.
            for i, x, y in zip(np.arange(len(X)), X, Y):
                y_hat, delta_psi, slack, loss = find_constraint(self.problem,
                                                                x, y, w)
                objective += slack
                psis.append(delta_psi)

                losses.append(loss)
                current_loss += loss
                if slack > 0:
                    positive_slacks += 1
            objective /= len(X)
            current_loss /= len(X)
            objective += np.sum(w ** 2) / self.C / 2.
            if positive_slacks == 0:
                print("No additional constraints")
                break
            if self.verbose > 0:
                print("iteration %d" % iteration)
                print("current loss: %f  positive slacks: %d, objective: %f" %
                      (current_loss, positive_slacks, objective))
            loss_curve.append(current_loss)
            all_psis.extend(psis)
            objective_curve.append(objective)
            w = self._solve_subgradient(psis)

            if self.verbose > 2:
                print(w)
        self.w = w
        print("final objective: %f" % objective_curve[-1])
        print("calls to inference: %d" % self.problem.inference_calls)
        if self.plot:
            plt.subplot(121, title="loss")
            plt.plot(loss_curve)
            plt.subplot(122, title="objective")
            plt.plot(objective_curve)
            plt.show()
