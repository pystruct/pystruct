######################
# (c) 2012 Andreas Mueller <amueller@ais.uni-bonn.de>
# ALL RIGHTS RESERVED.
#
# THIS IS UNPUBLISHED RESEARCH. DON'T USE WITHOUT AUTHOR CONSENT!
#

import numpy as np
import matplotlib.pyplot as plt
from structured_svm import StructuredSVM

from IPython.core.debugger import Tracer

tracer = Tracer()


class StupidLatentSVM(StructuredSVM):
    def fit(self, X, Y):
        w = np.ones(self.problem.size_psi) * 1e-5
        subsvm = StructuredSVM(self.problem, self.max_iter, self.C,
                self.check_constraints, verbose=self.verbose - 1, n_jobs=self.n_jobs)
        objectives = []
        ws = []
        H = Y
        Y = Y / self.problem.n_states_per_label
        # forget assignment of latent variables
        H = Y * self.problem.n_states_per_label

        for iteration in xrange(10):
            print("LATENT SVM ITERATION %d" % iteration)
            # find latent variables for ground truth:
            if iteration == 0:
                pass
            else:
                H_new = np.array([self.problem.latent(x, y, w) for x, y in zip(X, Y)])
                if np.all(H_new == H):
                    print("no changes in latent variables of ground truth. stopping.")
                    break
                H = H_new
            #X_wide = [np.repeat(x, self.problem.n_states_per_label, axis=1)
            #for x in X]
            subsvm.fit(X, H)
            H_hat = [self.problem.inference(x, subsvm.w) for x in X]
            inds = np.arange(len(H))
            for i, h, h_hat in zip(inds, H, H_hat):
                plt.matshow(h.reshape(x.shape[:-1]))
                plt.colorbar()
                plt.savefig("figures/h_%03d_%03d.png" % (iteration, i))
                plt.close()
                plt.matshow(h_hat.reshape(x.shape[:-1]))
                plt.colorbar()
                plt.savefig("figures/h_hat_%03d_%03d.png" % (iteration, i))
                plt.close()
            w = subsvm.w
            ws.append(w)
            objectives.append(subsvm.primal_objective_)
        self.w = w
        plt.figure()
        plt.plot(objectives)
        plt.show()
        tracer()


class LatentStructuredSVM(StructuredSVM):
    """Margin rescaled with l1 slack penalty."""
    def fit(self, X, Y):
        psi = self.problem.psi
        # we initialize with a small value so that loss-augmented inference
        # can give us something meaningful in the first iteration
        w = np.ones(self.problem.size_psi) * 1e-5
        n_samples = len(X)
        constraints = [[] for i in xrange(n_samples)]
        loss_curve = []
        objective_curve = []
        primal_objective_curve = []
        for iteration in xrange(self.max_iter):
            print("iteration %d" % iteration)
            new_constraints = 0
            current_loss = 0.
            primal_objective = 0.
            # loop over examples
            for i, x, y in zip(np.arange(len(X)), X, Y):
                # get latent variable for ground truth
                h = self.problem.latent(x, y, w)
                # get most violating constraint
                h_hat, y_hat = self.problem.loss_augmented_inference(x, y, w)
                loss = self.problem.loss(y, y_hat)
                delta_psi = psi(x, h, y) - psi(x, h_hat, y_hat)
                slack = loss - np.dot(w, delta_psi)
                if self.verbose > 1:
                    print("current slack: %f" % slack)

                primal_objective += slack

                if i < 5 and not iteration % 1:
                    plt.matshow(h.reshape(18, 18))
                    plt.colorbar()
                    plt.savefig("figures/h_%03d_%03d.png" % (iteration, i))
                    plt.close()
                    plt.matshow(h_hat.reshape(18, 18))
                    plt.colorbar()
                    plt.savefig("figures/h_hat_%03d_%03d.png" % (iteration, i))
                    plt.close()

                # recompute psi from previous constraints
                for j, con in enumerate(constraints[i]):
                    y_hat_old, _, loss_old = con
                    h_hat_old = self.problem.latent(x, y_hat_old, w)
                    delta_psi_old = psi(x, h, y) - psi(x, h_hat_old, y_hat_old)
                    constraints[i][j] = (y_hat_old, delta_psi_old, loss_old)

                #already_active = np.any([True for y_hat_, psi_, loss_ in
                    #constraints[i] if (y_hat == y_hat_).all()])
                already_active = False

                if self.check_constraints:
                    # "smart" but expensive stopping criterion
                    # check if most violated constraint is more violated
                    # than previous ones by more then eps.
                    # If it is less violated, inference was wrong/approximate
                    for con in constraints[i]:
                        # compute slack for old constraint
                        slack_tmp = con[2] - np.dot(w, con[1])
                        if self.verbose > 1:
                            print("slack old constraint: %f" % slack_tmp)
                        # if slack of new constraint is smaller or not
                        # significantly larger, don't add constraint.
                        # if smaller, complain about approximate inference.
                        if slack < slack_tmp:
                            print("bad inference!")
                            already_active = True
                            break
                        if (slack - slack_tmp) < 1e-5:
                            already_active = True
                            break

                current_loss += loss

                # if significant slack and constraint not active
                # this is a weaker check than the "check_constraints" one.
                if not already_active and slack > 1e-5:
                    delta_psi = psi(x, h, y) - psi(x, h_hat, y_hat)
                    constraints[i].append([y_hat, delta_psi, loss])
                    new_constraints += 1

            primal_objective /= len(X)
            current_loss /= len(X)
            primal_objective += np.sum(w ** 2) / self.C / 2.
            print("current loss: %f  new constraints: %d, primal obj: %f" %
                    (current_loss, new_constraints, primal_objective))
            loss_curve.append(current_loss)

            primal_objective_curve.append(primal_objective)
            if new_constraints == 0:
                print("no additional constraints found")
                break
            w, objective = self._solve_qp(constraints, n_samples)
            objective_curve.append(objective)
            if self.verbose > 0:
                print(w)
        self.w = w
        plt.subplot(131, title="loss")
        plt.plot(loss_curve)
        plt.subplot(132, title="objective")
        # the objective value should be monotonically decreasing
        # this is a maximization problem, to which we add more
        # and more constraints
        plt.plot(objective_curve)
        plt.subplot(133, title="primal objective")
        plt.plot(primal_objective_curve)
        plt.show()

    def predict(self, X):
        prediction = []
        for x in X:
            prediction.append(self.problem.inference(x, self.w)[1])
        return prediction
