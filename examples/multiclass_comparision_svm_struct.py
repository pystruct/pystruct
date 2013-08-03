"""
==================================================================
Comparing PyStruct and SVM-Struct for multi-class classification
==================================================================
This example compares the performance of pystruct and SVM^struct on a
multi-class problem.
For the example to work, you need to install SVM^multiclass and
set the path in this file.
We are not using SVM^python, as that would be much slower, and we would
need to implement our own model in a SVM^python compatible way.
Instead, we just call the SVM^multiclass binary.

This comparison is only meaningful in the sense that both libraries
use general structured prediction solvers to solve the task.
The specialized implementation of the Crammer-Singer SVM in LibLinear
is much faster than either one.

The plots are adjusted to disregard the time spend in writing
the data to the file for use with SVM^struct. As this time is
machine dependent, the plots are only approximate (unless you measure
that time for your machine and re-adjust)
"""

import tempfile
import os
from time import time

import numpy as np
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import fetch_mldata, load_iris, load_digits
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from pystruct.models import CrammerSingerSVMModel
from pystruct.learners import OneSlackSSVM

# please set the path to the svm-struct multiclass binaries here
svmstruct_path = "/home/local/lamueller/tools/svm_multiclass/"


class MultiSVM():
    """scikit-learn compatible interface for SVM^multi.

    Dumps the data to a file and calls the binary.
    """
    def __init__(self, C=1.):
        self.C = C

    def fit(self, X, y):
        self.model_file = tempfile.mktemp(suffix='.svm')
        train_data_file = tempfile.mktemp(suffix='.svm_dat')
        dump_svmlight_file(X, y + 1, train_data_file, zero_based=False)
        C = self.C * 100. * len(X)
        os.system(svmstruct_path + "svm_multiclass_learn -c %f %s %s"
                  % (C, train_data_file, self.model_file))

    def _predict(self, X, y=None):
        if y is None:
            y = np.ones(len(X))
        train_data_file = tempfile.mktemp(suffix='.svm_dat')

        dump_svmlight_file(X, y, train_data_file, zero_based=False)

        prediction_file = tempfile.mktemp(suffix='.out')
        os.system(svmstruct_path + "svm_multiclass_classify %s %s %s"
                  % (train_data_file, self.model_file, prediction_file))
        return np.loadtxt(prediction_file)

    def predict(self, X):
        return self._predict(X)[:, 0] - 1

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def decision_function(self, X):
        return self._predict(X)[:, 1:]


def eval_on_data(X, y, svm, Cs):
    accuracies, times = [], []
    for C in Cs:
        svm.C = C
        start = time()
        svm.fit(X, y)
        times.append(time() - start)
        accuracies.append(accuracy_score(y, svm.predict(X)))
    return accuracies, times


def plot_timings(times_svmstruct, times_pystruct, dataset="usps"):
    plt.figure()
    plt.figsize(4, 3)
    plt.plot(times_svmstruct, ":", label="SVM^struct", c='blue')
    plt.plot(times_pystruct, "-.", label="PyStruct", c='red')
    plt.xlabel("C")
    plt.xticks(np.arange(len(Cs)), Cs)
    plt.ylabel("learning time (s)")
    plt.legend(loc='best')
    plt.savefig("timings_%s.pdf" % dataset, bbox_inches='tight')


if __name__ == "__main__":
    Cs = 10. ** np.arange(-4, 1)
    multisvm = MultiSVM()
    svm = OneSlackSSVM(CrammerSingerSVMModel(tol=0.001))

    iris = load_iris()
    X, y = iris.data, iris.target

    accs_pystruct, times_pystruct = eval_on_data(X, y, svm, Cs=Cs)
    accs_svmstruct, times_svmstruct = eval_on_data(X, y, multisvm, Cs=Cs)

    # the adjustment of 0.01 is for the time spent writing the file, see above.
    plot_timings(np.array(times_svmstruct) - 0.01, times_pystruct,
                 dataset="iris")

    digits = load_digits()
    X, y = digits.data / 16., digits.target

    accs_pystruct, times_pystruct = eval_on_data(X, y, Cs=Cs)
    accs_svmstruct, times_svmstruct = eval_on_data(X, y, MultiSVM(), Cs=Cs)

    plot_timings(np.array(times_svmstruct) - 0.85, times_pystruct,
                 dataset="digits")

    digits = fetch_mldata("USPS")
    X, y = digits.data, digits.target.astype(np.int)

    accs_pystruct, times_pystruct = eval_on_data(X, y - 1, svm, Cs=Cs)
    accs_svmstruct, times_svmstruct = eval_on_data(X, y, multisvm, Cs=Cs)

    plot_timings(np.array(times_svmstruct) - 35, times_pystruct,
                 dataset="usps")
    plt.show()
