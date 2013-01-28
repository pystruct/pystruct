######################
# (c) 2012 Andreas Mueller <amueller@ais.uni-bonn.de>
# ALL RIGHTS RESERVED.
#
# DON'T USE WITHOUT AUTHOR CONSENT!
#

import numpy as np

from sklearn.externals.joblib import Parallel, delayed

from .cutting_plane_ssvm import StructuredSVM
from ..utils import inference, find_constraint


class LatentSSVM(StructuredSVM):
    def fit(self, X, Y, H_init=None):
        w = np.ones(self.problem.size_psi) * 1e-5
        subsvm = StructuredSVM(self.problem, self.max_iter, self.C,
                               self.check_constraints, verbose=self.verbose -
                               1, n_jobs=self.n_jobs,
                               break_on_bad=self.break_on_bad)
        #objectives = []
        constraints = None
        ws = []
        #Y = Y / self.problem.n_states_per_label
        if H_init is None:
            H_init = self.problem.init_latent(X, Y)
        #kmeans_init(X, Y, edges, self.problem.n_states_per_label)
        self.H_init_ = H_init
        H = H_init
        inds = np.arange(len(H))
        if False:
            import matplotlib.pyplot as plt
            for i, h in zip(inds, H):
                plt.matshow(h, vmin=0, vmax=self.problem.n_states - 1)
                plt.colorbar()
                plt.savefig("figures/h_init_%03d.png" % i)
                plt.close()

        for iteration in xrange(5):
            print("LATENT SVM ITERATION %d" % iteration)
            # find latent variables for ground truth:
            if iteration == 0:
                pass
            else:
                H_new = np.array([self.problem.latent(x, y, w)
                                  for x, y in zip(X, Y)])
                if np.all(H_new == H):
                    print("no changes in latent variables of ground truth."
                          " stopping.")
                    break
                print("changes in H: %d" % np.sum(H_new != H))

                # update constraints:
                constraints = [[] for i in xrange(len(X))]
                for sample, h, i in zip(subsvm.constraints_, H_new,
                                        np.arange(len(X))):
                    for constraint in sample:
                        const = find_constraint(self.problem, X[i], h, w,
                                                constraint[0])
                        y_hat, dpsi, _, loss = const
                        constraints[i].append([y_hat, dpsi, loss])
                H = H_new
            #if initialization weak?
            #if iteration == 0:
                #subsvm.max_iter = 10

            subsvm.fit(X, H, constraints=constraints)
            #if iteration == 0:
                #subsvm.max_iter = self.max_iter
            H_hat = Parallel(n_jobs=self.n_jobs, verbose=self.verbose - 2)(
                delayed(inference)(self.problem, x, subsvm.w) for x in X)
            inds = np.arange(len(H))
            if False:
                import matplotlib.pyplot as plt
                for i, h, h_hat in zip(inds, H, H_hat):
                    plt.matshow(h, vmin=0, vmax=self.problem.n_states - 1)
                    plt.colorbar()
                    plt.savefig("figures/h_%03d_%03d.png" % (iteration, i))
                    plt.close()
                    plt.matshow(h_hat, vmin=0, vmax=self.problem.n_states - 1)
                    plt.colorbar()
                    plt.savefig("figures/h_hat_%03d_%03d.png" % (iteration, i))
                    plt.close()
            w = subsvm.w
            ws.append(w)
            #objectives.append(subsvm.primal_objective_)
        self.w = w
        #plt.figure()
        #plt.plot(objectives)
        #plt.show()
