"""
===================
Latent Dynamics CRF
===================

Solving a 2d gid problem by introducing latent variable interactions.

"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

from pystruct.models import LatentGridCRF
#from pystruct.learners import LatentSSVM
from pystruct.learners import LatentSubgradientSSVM

import pystruct.toy_datasets as toy


def main():
    X, Y = toy.generate_crosses(n_samples=20, noise=5, n_crosses=1,
                                total_size=8)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5)
    n_labels = len(np.unique(Y_train))
    crf = LatentGridCRF(n_labels=n_labels, n_states_per_label=[1, 2],
                        inference_method='lp')
    #clf = LatentSSVM(model=crf, max_iter=500, C=1000., verbose=2,
                     #check_constraints=True, n_jobs=-1, break_on_bad=True,
                     #base_svm='1-slack', inference_cache=20, tol=.1)
    clf = LatentSubgradientSSVM(
        model=crf, max_iter=500, C=1000., verbose=2,
        n_jobs=-1, learning_rate=0.1, show_loss_every=10)
    clf.fit(X_train, Y_train)

    #for X_, Y_, H, name in [[X_train, Y_train, clf.H_init_, "train"],
                            #[X_test, Y_test, [None] * len(X_test), "test"]]:
    for X_, Y_, H, name in [[X_train, Y_train, [None] * len(X_test), "train"],
                            [X_test, Y_test, [None] * len(X_test), "test"]]:
        Y_pred = clf.predict(X_)
        i = 0
        loss = 0
        for x, y, h_init, y_pred in zip(X_, Y_, H, Y_pred):
            loss += np.sum(y != y_pred)
            fig, ax = plt.subplots(3, 2)
            ax[0, 0].matshow(y, vmin=0, vmax=crf.n_labels - 1)
            ax[0, 0].set_title("ground truth")
            ax[0, 1].matshow(np.argmax(x, axis=-1),
                             vmin=0, vmax=crf.n_labels - 1)
            ax[0, 1].set_title("unaries only")
            if h_init is None:
                ax[1, 0].set_visible(False)
            else:
                ax[1, 0].matshow(h_init, vmin=0, vmax=crf.n_states - 1)
                ax[1, 0].set_title("latent initial")
            ax[1, 1].matshow(crf.latent(x, y, clf.w),
                             vmin=0, vmax=crf.n_states - 1)
            ax[1, 1].set_title("latent final")
            ax[2, 0].matshow(crf.inference(x, clf.w),
                             vmin=0, vmax=crf.n_states - 1)
            ax[2, 0].set_title("prediction latent")
            ax[2, 1].matshow(y_pred,
                             vmin=0, vmax=crf.n_labels - 1)
            ax[2, 1].set_title("prediction")
            for a in ax.ravel():
                a.set_xticks(())
                a.set_yticks(())
            fig.savefig("data_%s_%03d.png" % (name, i), bbox_inches="tight")
            i += 1
        print("loss %s set: %f" % (name, loss))
    print(clf.w)

if __name__ == "__main__":
    main()
