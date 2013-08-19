"""
=================================
Comparing PyStruct and SVM-Struct
=================================
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

For SVM^struct, the plot show CPU time as reportet by SVM^struct.
For pystruct, the plot shows the time spent in the fit function
according to time.clock.

Both models have disabled constraint caching. With constraint caching,
SVM^struct is somewhat faster, but PyStruct doesn't gain anything.
"""

import tempfile
import os
from time import clock

import numpy as np
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import fetch_mldata, load_iris, load_digits
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

from pystruct.models import MultiClassClf
from pystruct.learners import OneSlackSSVM

# please set the path to the svm-struct multiclass binaries here
svmstruct_path = "/home/user/amueller/tools/svm_multiclass/"


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
        svmstruct_process = os.popen(svmstruct_path
                                     + "svm_multiclass_learn -w 3 -c %f %s %s"
                                     % (C, train_data_file, self.model_file))
        self.output_ = svmstruct_process.read().split("\n")
        self.runtime_ = float(self.output_[-4].split(":")[1])

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


def eval_on_data(X_train, y_train, X_test, y_test, svm, Cs):
    # evaluate a single svm using varying C
    accuracies, times = [], []
    for C in Cs:
        svm.C = C
        start = clock()
        svm.fit(X_train, y_train)
        if hasattr(svm, "runtime_"):
            times.append(svm.runtime_)
        else:
            times.append(clock() - start)
        accuracies.append(accuracy_score(y_test, svm.predict(X_test)))
    return accuracies, times


def plot_curves(curve_svmstruct, curve_pystruct, Cs, title="", filename=""):
    # plot nice graphs comparing a value for the two implementations
    plt.figure(figsize=(7, 4))
    plt.plot(curve_svmstruct, "--", label="SVM^struct", c='red', linewidth=3)
    plt.plot(curve_pystruct, "-.", label="PyStruct", c='blue', linewidth=3)
    plt.xlabel("C")
    plt.xticks(np.arange(len(Cs)), Cs)
    plt.legend(loc='best')
    plt.title(title)
    if filename:
        plt.savefig("%s" % filename, bbox_inches='tight')


def do_comparison(X_train, y_train, X_test, y_test, dataset):
    # evaluate both svms on a given datasets, generate plots
    Cs = 10. ** np.arange(-4, 1)
    multisvm = MultiSVM()
    svm = OneSlackSSVM(MultiClassClf(), tol=0.01)

    accs_pystruct, times_pystruct = eval_on_data(X_train, y_train, X_test,
                                                 y_test, svm, Cs=Cs)
    accs_svmstruct, times_svmstruct = eval_on_data(X_train, y_train,
                                                   X_test, y_test,
                                                   multisvm, Cs=Cs)

    plot_curves(times_svmstruct, times_pystruct, Cs=Cs,
                title="learning time (s) %s" % dataset,
                filename="times_%s.pdf" % dataset)
    plot_curves(accs_svmstruct, accs_pystruct, Cs=Cs,
                title="accuracy %s" % dataset,
                filename="accs_%s.pdf" % dataset)


def main():
    if not os.path.exists(svmstruct_path + "svm_multiclass_learn"):
        print("Please install SVM^multi and set the svmstruct_path variable "
              "to run this example.")
        return

    datasets = ['iris', 'digits']
    #datasets = ['iris', 'digits', 'usps', 'mnist']

    # IRIS
    if 'iris' in datasets:
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=0)
        do_comparison(X_train, y_train, X_test, y_test, "iris")

    # DIGITS
    if 'digits' in datasets:
        digits = load_digits()
        X, y = digits.data / 16., digits.target
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=0)
        do_comparison(X_train, y_train, X_test, y_test, "digits")

    # USPS
    if 'usps' in datasets:
        digits = fetch_mldata("USPS")
        X, y = digits.data, digits.target.astype(np.int) - 1
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=0)
        do_comparison(X_train, y_train, X_test, y_test, "USPS")

    # MNIST
    if 'mnist' in datasets:
        digits = fetch_mldata("MNIST original")
        X, y = digits.data / 255., digits.target.astype(np.int)
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]
        do_comparison(X_train, y_train, X_test, y_test, "MNIST")

    plt.show()


if __name__ == "__main__":
    main()
