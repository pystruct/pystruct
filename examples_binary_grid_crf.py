import numpy as np
import matplotlib.pyplot as plt

from crf import BinaryGridCRF
#from structured_perceptron import StructuredPerceptron
from structured_svm import StructuredSVM
#from structured_svm import SubgradientStructuredSVM

from toy_datasets import make_dataset_easy_latent

from IPython.core.debugger import Tracer
tracer = Tracer()


def main():
    #X, Y = make_dataset_blocks()
    #X, Y = make_dataset_checker()
    X, Y = make_dataset_easy_latent(n_samples=10)
    #X, Y = make_dataset_big_checker()
    crf = BinaryGridCRF()
    #clf = StructuredPerceptron(problem=crf, max_iter=100)
    clf = StructuredSVM(problem=crf, max_iter=200, C=100, verbose=0,
            check_constraints=False)
    #clf = SubgradientStructuredSVM(problem=crf, max_iter=2000, C=100)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    i = 0
    loss = 0
    for x, y, y_pred in zip(X, Y, Y_pred):
        loss += np.sum(y != y_pred)
        plt.subplot(221)
        plt.imshow(x[:, :, 0], interpolation='nearest')
        plt.subplot(222)
        plt.imshow(y, interpolation='nearest')
        plt.subplot(223)
        plt.imshow(x[:, :, 0] > 0, interpolation='nearest')
        plt.subplot(224)
        plt.imshow(y_pred, interpolation='nearest')
        plt.savefig("data_%03d.png" % i)
        plt.close()
        i += 1
    print("loss: %f" % loss)

if __name__ == "__main__":
    main()
