import numpy as np
import matplotlib.pyplot as plt

from latent_crf import LatentGridCRF
from latent_structured_svm import StupidLatentSVM

import toy_datasets as toy

from IPython.core.debugger import Tracer
tracer = Tracer()


def main():
    #X, Y = toy.generate_crosses_latent(n_samples=25, noise=10)
    #X, Y = toy.generate_crosses(n_samples=50, noise=10)
    #X, Y = toy.generate_easy(n_samples=50, noise=5)
    X, Y = toy.generate_xs(n_samples=25, noise=5)
    n_labels = 2
    #crf = LatentGridCRF(n_labels=n_labels, n_states_per_label=2,
                        #inference_method='dai')
    crf = LatentGridCRF(n_labels=n_labels, n_states_per_label=2,
                        inference_method='lp')
    clf = StupidLatentSVM(problem=crf, max_iter=50, C=10. ** 5, verbose=2,
                          check_constraints=True, n_jobs=12, break_on_bad=True)
    #clf = StupidLatentSVM(problem=crf, max_iter=50, C=1, verbose=2,
            #check_constraints=True, n_jobs=12)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)

    i = 0
    loss = 0
    for x, y, h_init, y_pred in zip(X, Y, clf.H_init_, Y_pred):
        y_pred = y_pred.reshape(x.shape[:2])
        loss += np.sum(y != y_pred / 2)
        if i > 100:
            continue
        fig, ax = plt.subplots(3, 2)
        ax[0, 0].matshow(y * crf.n_states_per_label,
                         vmin=0, vmax=crf.n_states - 1)
        ax[0, 0].set_title("ground truth")
        w_unaries_only = np.array([1, 1, 1, 1,
                                   0,
                                   0, 0,
                                   0, 0, 0,
                                   0, 0, 0, 0])
        unary_pred = crf.inference(x, w_unaries_only)
        ax[0, 1].matshow(unary_pred, vmin=0, vmax=crf.n_states - 1)
        ax[0, 1].set_title("unaries only")
        ax[1, 0].matshow(h_init, vmin=0, vmax=crf.n_states - 1)
        ax[1, 0].set_title("latent initial")
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
