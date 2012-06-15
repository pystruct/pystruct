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


class LatentStructuredSVM(StructuredSVM):
    """Margin rescaled with l1 slack penalty."""
    def fit(self, X, Y):
        psi = self.problem.psi
        # we initialize with a small value so that loss-augmented inference
        # can give us something meaningful in the first iteration
        w = np.ones(self.problem.size_psi) * 1e-5
        n_samples = len(X)
        constraints = [[] for i in xrange(n_samples)]
        psis = [[] for i in xrange(n_samples)]
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
                psis[i] = []
                # get latent variable for ground truth
                h = self.problem.latent(x, y, w)
                # get most violating constraint
                h_hat, y_hat = self.problem.loss_augmented_inference(x, y, w)
                loss = self.problem.loss(y, y_hat)
                #if i < 5 and not iteration % 10:
                    #plt.matshow(h.reshape(18, 18))
                    #plt.colorbar()
                    #plt.savefig("figures/h_%03d_%03d.png" % (iteration, i))
                    #plt.close()
                    #plt.matshow(h_hat.reshape(18, 18))
                    #plt.colorbar()
                    #plt.savefig("figures/h_hat_%03d_%03d.png" % (iteration, i))
                    #plt.close()

                # recompute psi from previous constraints
                for y_hat_old, loss_old in constraints[i]:
                    h_hat_old = self.problem.latent(x, y_hat_old, w)
                    delta_psi_old = psi(x, h, y) - psi(x, h_hat_old, y_hat_old)
                    psis[i].append(delta_psi_old)

                already_active = np.any([True for y_hat_, psi_, loss_ in
                    constraints[i] if (y_hat == y_hat_).all()])

                delta_psi = psi(x, y) - psi(x, y_hat)
                slack = loss - np.dot(w, delta_psi)
                if self.verbose > 1:
                    print("current slack: %f" % slack)

                primal_objective += slack

                if self.check_constraints:
                    # "smart" but expensive stopping criterion
                    # check if most violated constraint is more violated
                    # than previous ones by more then eps.
                    # If it is less violated, inference was wrong/approximate
                    for con, dpsi in zip(constraints[i], psis[i]):
                        # compute slack for old constraint
                        h_tmp = self.problem.latent(x, con[0], w)
                        dpsi_tmp = psi(x, h, y) - psi(x, h_tmp, con[0])
                        loss_tmp = self.problem.loss(y, con[0])
                        slack_tmp = loss_tmp - np.dot(w, dpsi_tmp)
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
                    constraints[i].append([y_hat, loss])
                    delta_psi = psi(x, h, y) - psi(x, h_hat, y_hat)
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
            w = self._solve_qp(constraints, psis, n_samples)
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
