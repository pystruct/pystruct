######################
# (c) 2012 Andreas Mueller <amueller@ais.uni-bonn.de>
# ALL RIGHTS RESERVED.
#
# THIS IS UNPUBLISHED RESEARCH. DON'T USE WITHOUT AUTHOR CONSENT!
#

import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from sklearn.cluster import KMeans

from structured_svm import StructuredSVM, inference, find_constraint

from IPython.core.debugger import Tracer

tracer = Tracer()


def kmeans_init(X, Y, n_states_per_label=2):
    n_labels = X[0].shape[-1]
    shape = Y[0].shape
    gx, gy = np.ogrid[:shape[0], :shape[1]]
    all_feats = []
    # iterate over samples
    for x, y in zip(X, Y):
        # first, get neighbor counts from nodes
        labels = np.zeros((shape[0], shape[1], n_labels),
                          dtype=np.int)
        labels[gx, gy, y] = 1
        neighbors = np.zeros((y.shape[0], y.shape[1], n_labels))
        neighbors[1:, :, :] += labels[:-1, :, :]
        neighbors[:-1, :, :] += labels[1:, :, :]
        neighbors[:, 1:, :] += labels[:, :-1, :]
        neighbors[:, :-1, :] += labels[:, 1:, :]
        # normalize (for borders)
        neighbors /= neighbors.sum(axis=-1)[:, :, np.newaxis]
        # add unaries
        #features = np.dstack([x, neighbors])
        features = neighbors
        all_feats.append(features.reshape(-1, features.shape[-1]))
    all_feats = np.vstack(all_feats)
    # states (=clusters) will be saved in H
    H = np.zeros_like(Y, dtype=np.int)
    km = KMeans(n_clusters=n_states_per_label)
    # for each state, run k-means over whole dataset
    for label in np.arange(n_labels):
        indicator = Y.ravel() == label
        f = all_feats[indicator]
        states = km.fit_predict(f)
        H.ravel()[indicator] = states + label * n_states_per_label
    return H


class StupidLatentSVM(StructuredSVM):
    def fit(self, X, Y):
        w = np.ones(self.problem.size_psi) * 1e-5
        subsvm = StructuredSVM(self.problem, self.max_iter, self.C,
                               self.check_constraints, verbose=self.verbose -
                               1, n_jobs=self.n_jobs,
                               break_on_bad=self.break_on_bad)
        #objectives = []
        constraints = None
        ws = []
        #Y = Y / self.problem.n_states_per_label
        H = kmeans_init(X, Y, self.problem.n_states_per_label)
        self.H_init_ = H
        inds = np.arange(len(H))
        for i, h in zip(inds, H):
            plt.matshow(h, vmin=0, vmax=self.problem.n_states - 1)
            plt.colorbar()
            plt.savefig("figures/h_init_%03d.png" % i)
            plt.close()

        for iteration in xrange(10):
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

            #subsvm.fit(X, H, constraints=new_constraints)
            #constraints = subsvm.constraints_
            subsvm.fit(X, H, constraints=constraints)
            #if iteration == 0:
                #subsvm.max_iter = self.max_iter
            H_hat = Parallel(n_jobs=self.n_jobs)(delayed(inference)
                                                 (self.problem, x, subsvm.w)
                                                 for x in X)
            inds = np.arange(len(H))
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
