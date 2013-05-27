"""
==================================
Learning interactions on a 2d grid
==================================
Simple pairwise model with arbitrary interactions on a 4-connected grid.
"""

import numpy as np
import matplotlib.pyplot as plt

from pystruct.models import DirectionalGridCRF
#from pystruct.models import GridCRF
#from structured_perceptron import StructuredPerceptron
import pystruct.learners as ssvm
import pystruct.toy_datasets as toy


def main():
    X, Y = toy.generate_blocks_multinomial(noise=2, n_samples=20, seed=0)
    #X, Y = toy.generate_crosses_explicit(n_samples=50, noise=10)
    #X, Y = toy.generate_easy_explicit(n_samples=25, noise=10)
    #X, Y = toy.generate_checker_multinomial(n_samples=20)
    n_labels = len(np.unique(Y))
    crf = DirectionalGridCRF(n_states=n_labels, inference_method="lp",
                             neighborhood=4)
    clf = ssvm.OneSlackSSVM(model=crf, max_iter=1000, C=100, verbose=2,
                            check_constraints=True, n_jobs=-1,
                            inference_cache=100, inactive_window=50, tol=.1)
    #clf = ssvm.StructuredSVM(model=crf, max_iter=100, C=100, verbose=3,
                             #check_constraints=True, n_jobs=12)
    #clf = StructuredPerceptron(model=crf, max_iter=1000, verbose=10)
    #clf = ssvm.SubgradientSSVM(model=crf, max_iter=50, C=100,
                                        #verbose=10, momentum=.9,
                                        #learning_rate=0.04,
                                        #n_jobs=-1)
    clf.fit(X, Y)
    Y_pred = np.array(clf.predict(X))

    np.set_printoptions(suppress=True)  # suppress scientific notation
    print(clf.w)

    i = 0
    loss = 0
    for x, y, y_pred in zip(X, Y, Y_pred):
        y_pred = y_pred.reshape(x.shape[:2])
        loss += np.sum(y != y_pred)
        #if i > 10:
            #continue
        fig, plots = plt.subplots(1, 4)
        plots[0].matshow(y)
        plots[0].set_title("gt")
        plots[1].matshow(np.argmax(x, axis=-1))
        plots[1].set_title("unaries only")
        plots[2].matshow(y_pred)
        plots[2].set_title("prediction")
        loss_augmented = clf.model.loss_augmented_inference(x, y, clf.w)
        loss_augmented = loss_augmented.reshape(y.shape)
        plots[3].matshow(loss_augmented)
        plots[3].set_title("loss augmented")
        for p in plots:
            p.set_xticks(())
            p.set_yticks(())
        fig.savefig("data_%03d.png" % i)
        #plt.close(fig)
        i += 1
    print("loss: %f" % loss)
    print("complete loss: %f" % np.sum(Y != Y_pred))

if __name__ == "__main__":
    main()
