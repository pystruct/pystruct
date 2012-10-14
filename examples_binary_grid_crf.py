import numpy as np
import matplotlib.pyplot as plt

from crf import BinaryGridCRF
#from structured_perceptron import StructuredPerceptron
from structured_svm import StructuredSVM
#from structured_svm import SubgradientStructuredSVM

from toy_datasets import generate_easy

from IPython.core.debugger import Tracer
tracer = Tracer()


def main():
    #X, Y = generate_blocks(n_samples=10)
    #X, Y = generate_easy(n_samples=10)
    #X, Y = generate_big_checker()
    crf = BinaryGridCRF()
    #clf = StructuredPerceptron(problem=crf, max_iter=100)
    clf = StructuredSVM(problem=crf, max_iter=200, C=100, verbose=10,
            check_constraints=True)
    #clf = SubgradientStructuredSVM(problem=crf, max_iter=2000, C=100,
            #verbose=10, momentum=.9)
    clf.fit(X, Y)
    Y_pred = clf.predict(X)
    i = 0
    loss = 0
    for x, y, y_pred in zip(X, Y, Y_pred):
        loss += np.sum(y != y_pred)
        fig, plots = plt.subplots(2, 2)
        plots[0, 0].set_title("x")
        plots[0, 0].matshow(-x[:, :, 0])
        plots[0, 1].set_title("y")
        plots[0, 1].matshow(y)
        plots[1, 0].set_title("unaries")
        w_unaries = np.zeros(2)
        w_unaries[0] = 1.
        y_unaries = crf.inference(x, w_unaries)
        plots[1, 0].matshow(y_unaries)
        plots[1, 1].set_title("full crf")
        plots[1, 1].matshow(y_pred)
        for plot in plots.ravel():
            plot.set_axis_off()
        fig.savefig("data_%03d.png" % i)
        plt.close(fig)
        i += 1
    print("loss: %f" % loss)

if __name__ == "__main__":
    main()
