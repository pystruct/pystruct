# simply linear binary SVM with soft margin

import numpy as np
import cvxopt
import cvxopt.solvers
from IPython.core.debugger import Tracer
tracer = Tracer()


class SVM(object):
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # identity matrix for quadratic penalty
        # add one for bias but don't penalize
        eye = cvxopt.spmatrix(1.0, np.arange(n_features + 1),
                np.arange(n_features + 1))
        eye[-1, -1] = 0
        # append dummy feature for bias to data
        X = np.hstack([X, np.ones(X.shape[0])[:, np.newaxis]])
        zeros_features = cvxopt.matrix(np.zeros(n_features + 1))
        margin = cvxopt.matrix(-np.ones(n_samples))

        Xy = cvxopt.matrix(-X * y[:, np.newaxis])
        result = cvxopt.solvers.qp(P=eye, q=zeros_features, G=Xy,
                h=margin)
        params = np.array(result['x'])
        self.w = params[:-1]
        self.b = params[-1]

    def predict(self, X, y):
        pass

    def decision_function(self, X, y):
        pass


def main():
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    #from sklearn.preprocessing import Scaler
    X, y = make_blobs(centers=2)
    X += 1
    svm = SVM()
    y = 2 * y - 1
    svm.fit(X, y)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.scatter(0, 0, c='black', s=50)
    offset = np.array([svm.w[0] * -svm.b, svm.w[1] * -svm.b])
    plt.plot([0, 0], [offset[0], offset[1]])
    plt.show()
    tracer()

if __name__ == "__main__":
    main()
