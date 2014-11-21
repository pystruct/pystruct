import numpy as np
from scipy import sparse

from sklearn.cross_validation import train_test_split
from scipy.sparse.csgraph import minimum_spanning_tree

from pystruct.learners import SubgradientSSVM
from pystruct.models import GraphCRF


def make_random_trees(n_samples=50, n_nodes=100, n_states=7, n_features=10):
    crf = GraphCRF(inference_method='max-product', n_states=n_states,
                   n_features=n_features)
    weights = np.random.randn(crf.size_joint_feature)
    X, y = [], []
    for i in range(n_samples):
        distances = np.random.randn(n_nodes, n_nodes)
        features = np.random.randn(n_nodes, n_features)
        tree = minimum_spanning_tree(sparse.csr_matrix(distances))
        edges = np.c_[tree.nonzero()]
        X.append((features, edges))
        y.append(crf.inference(X[-1], weights))

    return X, y, weights


X, y, weights = make_random_trees(n_nodes=1000)

X_train, X_test, y_train, y_test = train_test_split(X, y)

#tree_model = MultiLabelClf(edges=tree, inference_method=('ogm', {'alg': 'dyn'}))
tree_model = GraphCRF(inference_method='max-product')

tree_ssvm = SubgradientSSVM(tree_model, max_iter=4, C=1, verbose=10)

print("fitting tree model...")
tree_ssvm.fit(X_train, y_train)

print("Training loss tree model: %f" % tree_ssvm.score(X_train, y_train))
print("Test loss tree model: %f" % tree_ssvm.score(X_test, y_test))
