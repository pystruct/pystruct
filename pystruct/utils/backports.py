from sklearn.utils import check_arrays
from sklearn.cross_validation import ShuffleSplit


# port from scikit-learn for old scikit-learn versions

def train_test_split(*arrays, **options):
    """Split arrays or matrices into random train and test subsets

    Quick utility that wraps calls to ``check_arrays`` and
    ``next(iter(ShuffleSplit(n_samples)))`` and application to input
    data into a single call for splitting (and optionally subsampling)
    data in a oneliner.

    Parameters
    ----------
    *arrays : sequence of arrays or scipy.sparse matrices with same shape[0]
        Python lists or tuples occurring in arrays are converted to 1D numpy
        arrays.

    test_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples. If None,
        the value is automatically set to the complement of the train size.
        If train size is also None, test size is set to 0.25.

    train_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.

    dtype : a numpy dtype instance, None by default
        Enforce a specific dtype.

    Returns
    -------
    splitting : list of arrays, length=2 * len(arrays)
        List containing train-test split of input array.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cross_validation import train_test_split
    >>> a, b = np.arange(10).reshape((5, 2)), range(5)
    >>> a
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7],
           [8, 9]])
    >>> list(b)
    [0, 1, 2, 3, 4]

    >>> a_train, a_test, b_train, b_test = train_test_split(
    ...     a, b, test_size=0.33, random_state=42)
    ...
    >>> a_train
    array([[4, 5],
           [0, 1],
           [6, 7]])
    >>> b_train
    array([2, 0, 3])
    >>> a_test
    array([[2, 3],
           [8, 9]])
    >>> b_test
    array([1, 4])

    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    test_size = options.pop('test_size', None)
    train_size = options.pop('train_size', None)
    random_state = options.pop('random_state', None)
    options['sparse_format'] = 'csr'

    if test_size is None and train_size is None:
        test_size = 0.25

    arrays = check_arrays(*arrays, **options)
    n_samples = arrays[0].shape[0]
    try:
        cv = ShuffleSplit(n_samples, test_size=test_size,
                          train_size=train_size,
                          random_state=random_state,
                          indices=True)
    except TypeError:
        if test_size > 1:
            test_size /= n_samples
            if train_size is not None:
                train_size /= n_samples
        cv = ShuffleSplit(n_samples, test_fraction=test_size,
                          train_fraction=train_size,
                          random_state=random_state,
                          indices=True)
    train, test = next(iter(cv))
    splitted = []
    for a in arrays:
        splitted.append(a[train])
        splitted.append(a[test])
    return splitted

train_test_split.__test__ = False  # to avoid a pb with nosetests
