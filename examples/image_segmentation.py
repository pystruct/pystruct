"""
===========================================
Semantic Image Segmentation on Pascal VOC
===========================================
This example demonstrates learning a superpixel CRF
for semantic image segmentation.
To run the experiment, please download the pre-processed data from:

The data consists of superpixels, unary potentials, and the connectivity
structure of the superpixels.
The unary potentials were originally provided by Philipp Kraehenbuehl:

The superpixels were extracted using SLIC.
The code for generating the connectivity graph and edge features will be made
public soon.

This example does not contain the proper evaluation on pixel level, as that
would need the Pascal VOC 2010 dataset.
"""
import numpy as np
import cPickle

from pystruct import learners
import pystruct.models as crfs
from pystruct.utils import SaveLogger


data_train = cPickle.load(open("data_train.pickle"))
C = 0.01

n_states = 21
print("number of samples: %s" % len(data_train.X))
class_weights = 1. / np.bincount(np.hstack(data_train.Y))
class_weights *= 21. / np.sum(class_weights)
print(class_weights)

model = crfs.EdgeFeatureGraphCRF(n_states=n_states,
                                 n_features=data_train.X[0][0].shape[1],
                                 inference_method='qpbo',
                                 class_weight=class_weights,
                                 n_edge_features=3,
                                 symmetric_edge_features=[0, 1],
                                 antisymmetric_edge_features=[2])

experiment_name = "edge_features_one_slack_trainval_%f" % C

ssvm = learners.NSlackSSVM(
    model, verbose=2, C=C, max_iter=100000, n_jobs=-1,
    tol=0.0001, show_loss_every=5,
    logger=SaveLogger(experiment_name + ".pickle", save_every=100),
    inactive_threshold=1e-3, inactive_window=10, batch_size=100)
ssvm.fit(data_train.X, data_train.Y)

data_val = cPickle.load(open("data_val.pickle"))
y_pred = ssvm.predict(data_val.X)

# we throw away void superpixels and flatten everything
y_pred, y_true = np.hstack(y_pred), np.hstack(data_val.Y)
y_pred = y_pred[y_true != 255]
y_true = y_true[y_true != 255]

print("Score on validation set: %f" % np.mean(y_true == y_pred))
