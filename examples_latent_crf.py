import numpy as np
import matplotlib.pyplot as plt

from latent_crf import LatentGridCRF
from latent_structured_svm import StupidLatentSVM

import toy_datasets as toy

from IPython.core.debugger import Tracer
tracer = Tracer()


def main():
    X, Y = toy.generate_crosses_latent(n_samples=5, noise=10)
    n_labels = 2
    crf = LatentGridCRF(n_labels=n_labels, n_states_per_label=2,
                        inference_method='dai')
    clf = StupidLatentSVM(problem=crf, max_iter=100, C=10 ** 8, verbose=20,
            check_constraints=True)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)

    i = 0
    loss = 0
    tracer()
    for x, y, y_pred in zip(X, Y, Y_pred):
        y_pred = y_pred.reshape(x.shape[:2])
        loss += np.sum(y / 2 != y_pred / 2)
        if i > 20:
            continue
        plt.subplot(131)
        plt.imshow(y, interpolation='nearest')
        plt.colorbar()
        plt.subplot(132)
        plt.imshow(np.argmin(x, axis=2), interpolation='nearest')
        plt.colorbar()
        plt.subplot(133)
        plt.imshow(y_pred, interpolation='nearest')
        plt.colorbar()
        plt.savefig("data_%03d.png" % i)
        plt.close()
        i += 1
    print("loss: %f" % loss)

if __name__ == "__main__":
    main()
