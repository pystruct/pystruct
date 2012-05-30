import numpy as np
#import matplotlib.pyplot as plt

from crf import BinaryGridCRF
from structured_perceptron import StructuredPerceptron

from IPython.core.debugger import Tracer
tracer = Tracer()


def make_dataset_blocks():
    Y = np.ones((20, 10, 12))
    Y[:, :, :6] = -1
    X = Y + 1.5 * np.random.normal(size=Y.shape)
    Y = (Y > 0).astype(np.int32)
    return X, Y


def make_dataset_checker():
    Y = np.ones((20, 11, 13))
    Y[:, ::2, ::2] = -1
    Y[:, 1::2, 1::2] = -1
    X = Y + 1.5 * np.random.normal(size=Y.shape)
    Y = (Y > 0).astype(np.int32)
    return X, Y


def make_dataset_big_checker():
    _, Y_small = make_dataset_checker()
    Y = Y_small.repeat(3, axis=1).repeat(3, axis=2)
    X = Y + 0.5 * np.random.normal(size=Y.shape)
    Y = (Y > 0).astype(np.int32)
    return X, Y


def main():
    X, Y = make_dataset_blocks()
    #X, Y = make_dataset_checker()
    #X, Y = make_dataset_big_checker()
    crf = BinaryGridCRF()
    clf = StructuredPerceptron(probem=crf)
    clf.fit(X, Y)

    #for x, y in zip(X, Y):
        #unaries = (-100 * unary_param * np.dstack([x,
            #-x])).astype(np.int32).copy("C")
        #y_pred = binary_grid(unaries, pairwise)
        #plt.subplot(221)
        #plt.imshow(x, interpolation='nearest')
        #plt.subplot(222)
        #plt.imshow(y, interpolation='nearest')
        #plt.subplot(223)
        #plt.imshow(x > 0, interpolation='nearest')
        #plt.subplot(224)
        #plt.imshow(y_pred, interpolation='nearest')
        #plt.savefig("data_%03d.png" % i)
        #plt.close()
        #i += 1

if __name__ == "__main__":
    main()
