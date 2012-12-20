import numpy as np
import matplotlib.pyplot as plt

import latent_crf
from latent_structured_svm import LatentSSVM

import toy_datasets as toy

from IPython.core.debugger import Tracer
tracer = Tracer()


def main():
    #X, Y = toy.generate_crosses_latent(n_samples=25, noise=10)
    #X, Y = toy.generate_crosses(n_samples=20, noise=10)
    #X, Y = toy.generate_crosses(n_samples=2, noise=5, n_crosses=1,
                                #total_size=8)
    #X, Y = toy.generate_easy(n_samples=10, noise=5, box_size=2, total_size=6)
    X, Y = toy.generate_bars(n_samples=25, noise=8, bars_size=3, total_size=6)
    #X, Y = toy.generate_xs(n_samples=25, noise=5)
    n_labels = 3
    #crf = latent_crf.LatentGridCRF(n_labels=n_labels, n_states_per_label=4,
                                   #inference_method='lp')
    crf = latent_crf.LatentDirectionalGridCRF(n_labels=n_labels,
                                              n_states_per_label=3,
                                              inference_method='lp')
    clf = LatentSSVM(problem=crf, max_iter=50, C=10. ** 5, verbose=2,
                     check_constraints=True, n_jobs=12, break_on_bad=True,
                     plot=True)
    #clf = LatentSVM(problem=crf, max_iter=50, C=1, verbose=2,
            #check_constraints=True, n_jobs=12)
    clf.fit(X, Y)
    # generate a test set
    X, Y = toy.generate_bars(n_samples=25, noise=8, bars_size=3, total_size=6,
                             random_seed=10)
    Y_pred = clf.predict(X)

    i = 0
    loss = 0
    for x, y, h_init, y_pred in zip(X, Y, clf.H_init_, Y_pred):
        loss += np.sum(y != y_pred / crf.n_states_per_label)
        if i > 100:
            continue
        fig, ax = plt.subplots(3, 2)
        ax[0, 0].matshow(y * crf.n_states_per_label,
                         vmin=0, vmax=crf.n_states - 1)
        ax[0, 0].set_title("ground truth")
        #w_unaries_only = np.array([1, 1, 1, 1,
                                   #0,
                                   #0, 0,
                                   #0, 0, 0,
                                   #0, 0, 0, 0])
        w_unaries_only = np.zeros_like(clf.w)
        size_unaries = len(crf.get_unary_weights(clf.w))
        w_unaries_only[:size_unaries] = 1
        unary_pred = crf.inference(x, w_unaries_only)
        ax[0, 1].matshow(unary_pred, vmin=0, vmax=crf.n_states - 1)
        ax[0, 1].set_title("unaries only")
        #ax[1, 0].matshow(h_init, vmin=0, vmax=crf.n_states - 1)
        #ax[1, 0].set_title("latent initial")
        ax[1, 1].matshow(crf.latent(x, y, clf.w),
                         vmin=0, vmax=crf.n_states - 1)
        ax[1, 1].set_title("latent final")
        ax[2, 0].matshow(y_pred, vmin=0, vmax=crf.n_states - 1)
        ax[2, 0].set_title("prediction")
        ax[2, 1].matshow((y_pred // crf.n_states_per_label)
                         * crf.n_states_per_label,
                         vmin=0, vmax=crf.n_states - 1)
        ax[2, 1].set_title("prediction")
        for a in ax.ravel():
            a.set_xticks(())
            a.set_yticks(())
        fig.savefig("data_%03d.png" % i, bbox_inches="tight")
        i += 1
    print("loss: %f" % loss)
    print(clf.w)

if __name__ == "__main__":
    main()
