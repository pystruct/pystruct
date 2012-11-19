######################
# (c) 2012 Andreas Mueller <amueller@ais.uni-bonn.de>
# ALL RIGHTS RESERVED.
#
# THIS IS UNPUBLISHED RESEARCH. DON'T USE WITHOUT AUTHOR CONSENT!
#

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from structured_svm import StructuredSVM, inference

from IPython.core.debugger import Tracer

tracer = Tracer()


class StupidLatentSVM(StructuredSVM):
    def fit(self, X, Y):
        w = np.ones(self.problem.size_psi) * 1e-5
        subsvm = StructuredSVM(self.problem, self.max_iter, self.C,
                self.check_constraints, verbose=self.verbose - 1,
                n_jobs=self.n_jobs)
        objectives = []
        ws = []
        H = Y
        Y = Y / self.problem.n_states_per_label
        # forget assignment of latent variables
        H = Y * self.problem.n_states_per_label
        # randomize!
        H += np.random.randint(2, size=H.shape)
        inds = np.arange(len(H))
        for i, h in zip(inds, H):
            plt.matshow(h, vmin=0, vmax=self.n_states)
            plt.colorbar()
            plt.savefig("figures/h_0000_init_%03d.png" % (i))
            plt.close()

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
            H_hat = Parallel(n_jobs=self.n_jobs)(delayed(inference)(self.problem, x, subsvm.w) for x in X)
            inds = np.arange(len(H))
            for i, h, h_hat in zip(inds, H, H_hat):
                plt.matshow(h, vmin=0, vmax=self.n_states)
                plt.colorbar()
                plt.savefig("figures/h_%03d_%03d.png" % (iteration, i))
                plt.close()
                plt.matshow(h_hat, vmin=0, vmax=self.n_states)
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
