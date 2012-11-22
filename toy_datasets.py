import numpy as np
from IPython.core.debugger import Tracer

tracer = Tracer()


#### binary
def generate_blocks(n_samples=10, noise=1.5):
    np.random.seed(0)
    Y = np.ones((n_samples, 10, 12))
    Y[:, :, :6] = -1
    #Y = np.ones((n_samples, 3, 4))
    #Y[:, :, :2] = -1
    X = Y + noise * np.random.normal(size=Y.shape)
    X = np.c_['3,4,0', -X, X]
    Y = (Y > 0).astype(np.int32)
    return X, Y


def generate_checker(n_samples=10, noise=1.5):
    np.random.seed(0)
    Y = np.ones((n_samples, 11, 13))
    Y[:, ::2, ::2] = -1
    Y[:, 1::2, 1::2] = -1
    X = Y + noise * np.random.normal(size=Y.shape)
    X = np.c_['3,4,0', -X, np.zeros_like(X)]
    Y = (Y > 0).astype(np.int32)
    return X, Y


def generate_big_checker(n_samples=10, noise=0.5):
    np.random.seed(0)
    y_small = np.ones((5, 5), dtype=np.int32)
    y_small[::2, ::2] = -1
    y_small[1::2, 1::2] = -1
    y = y_small.repeat(3, axis=0).repeat(3, axis=1)
    Y = np.repeat(y[np.newaxis, :, :], n_samples, axis=0)
    X = Y + noise * np.random.normal(size=Y.shape)
    Y = (Y < 0).astype(np.int32)
    # make unaries
    X = np.r_['-1, 4,0', X, -X].copy("C")
    return X, Y


def generate_easy(n_samples=5, noise=5):
    size = 10
    np.random.seed(0)
    Y = np.ones((n_samples, size, size))
    for i in xrange(n_samples):
        for j in xrange(2):
            t, l = np.random.randint(size - 3, size=2)
            Y[i, t:t + 3, l:l + 3] = -1
    X = Y  # + noise * np.random.normal(size=Y.shape)
    X = np.c_['3,4,0', -X, np.zeros_like(X)]
    for x in X:
        flips = np.random.randint(size, size=[noise, 2])
        x[flips[:, 0], flips[:, 1], 0] = 1 - 2 * np.random.randint(2,
                                                                   size=noise)
    Y = (Y > 0).astype(np.int32)
    return X * 10, Y


def generate_crosses(n_samples=5, noise=30):
    np.random.seed(0)
    size = 8
    Y = np.ones((n_samples, size, size), dtype=np.int)
    for i in xrange(n_samples):
        for j in xrange(2):
            t, l = np.random.randint(size - 2, size=2)
            Y[i, t + 1, l:l + 3] = -1
            Y[i, t:t + 3, l + 1] = -1
    X = np.c_['3,4,0', -Y, np.zeros_like(Y)].astype(np.float)
    #flip random bits
    for x in X:
        flips = np.random.randint(size, size=[noise, 2])
        x[flips[:, 0], flips[:, 1], 0] = 1 - 2 * np.random.randint(2,
                                                                   size=noise)
    Y = (Y > 0).astype(np.int32)
    return X, Y


#### Multinomial
def generate_blocks_multinomial(n_samples=20, noise=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    Y = np.zeros((n_samples, 10, 12, 3))
    Y[:, :, :4, 0] = 1
    Y[:, :, 4:8, 1] = 1
    Y[:, :, 8:16, 2] = 1
    X = Y + noise * np.random.normal(size=Y.shape)
    Y = np.argmax(Y, axis=3).astype(np.int32)
    return X, Y


def generate_checker_multinomial(n_samples=20, noise=1.5):
    Y = -np.ones((n_samples, 10, 12, 3))
    Y[:, ::2, ::2, 0] = 1
    Y[:, 1::2, 1::2, 1] = 1
    Y[:, :, :, 2] = 0
    X = Y + noise * np.random.normal(size=Y.shape)
    Y = np.argmax(Y, axis=3).astype(np.int32)
    return X, Y


def generate_big_checker_extended(n_samples=20, noise=.3):
    y_small = np.zeros((6, 6), dtype=np.int32)
    y_small[::2, ::2] = 2
    y_small[1::2, 1::2] = 2
    y = y_small.repeat(3, axis=0).repeat(3, axis=1)
    y[1::3, 1::3] = 1
    y[1::6, 1::6] = 3
    y[4::6, 4::6] = 3
    Y = np.repeat(y[np.newaxis, :, :], n_samples, axis=0)
    X_shape = list(Y.shape)
    X_shape.append(4)
    X = np.zeros(X_shape)
    gx, gy, gz = np.mgrid[:Y.shape[0], :Y.shape[1], :Y.shape[2]]
    X[gx, gy, gz, Y] = 1
    X = X + noise * np.random.normal(size=X.shape)
    return X * 100., Y


def generate_easy_explicit(n_samples=5, noise=30):
    np.random.seed(0)
    Y = np.zeros((n_samples, 18, 18), dtype=np.int)
    for i in xrange(n_samples):
        for j in xrange(3):
            t, l = np.random.randint(15, size=2)
            Y[i, t:t + 3, l:l + 3] = 1
            Y[i, t + 1, l + 1] = 2
    Y_flips = Y.copy()
    #flip random bits
    for y in Y_flips:
        flips = np.random.randint(18, size=[noise, 2])
        y[flips[:, 0], flips[:, 1]] = np.random.randint(3, size=noise)
    X = np.zeros((n_samples, 18, 18, 3))
    ix, iy, iz = np.ogrid[:X.shape[0], :X.shape[1], :X.shape[2]]
    X[ix, iy, iz, Y_flips] = 1
    return X, Y


def generate_crosses_explicit(n_samples=5, noise=30):
    np.random.seed(0)
    size = 8
    Y = np.zeros((n_samples, size, size), dtype=np.int)
    for i in xrange(n_samples):
        for j in xrange(2):
            t, l = np.random.randint(size - 2, size=2)
            Y[i, t + 1, l:l + 3] = 1
            Y[i, t:t + 3, l + 1] = 1
            Y[i, t + 1, l + 1] = 2
    # don't distinguish between 1 an 2 in X
    Y_flips = (Y.copy() != 0).astype(np.int)
    #flip random bits
    for y in Y_flips:
        flips = np.random.randint(size, size=[noise, 2])
        y[flips[:, 0], flips[:, 1]] = np.random.randint(2, size=noise)
    X = np.zeros((n_samples, size, size, 3))
    ix, iy, iz = np.ogrid[:X.shape[0], :X.shape[1], :X.shape[2]]
    X[ix, iy, iz, Y_flips] = 1
    X[ix, iy, iz, 2 * Y_flips] = 1
    #Y = (Y != 0).astype(np.int)
    #X = X[:, :, :, :2]
    return X, Y


def generate_crosses_latent(n_samples=5, noise=30):
    # X knows two states, Y knows four.
    np.random.seed(0)
    size = 8
    Y = np.zeros((n_samples, size, size), dtype=np.int)
    for i in xrange(n_samples):
        for j in xrange(2):
            t, l = np.random.randint(size - 2, size=2)
            Y[i, t + 1, l:l + 3] = 2
            Y[i, t:t + 3, l + 1] = 2
            Y[i, t + 1, l + 1] = 3
    # don't distinguish between 1 an 2 in X
    Y_flips = (Y.copy() != 0).astype(np.int)
    #flip random bits
    for y in Y_flips:
        flips = np.random.randint(size, size=[noise, 2])
        y[flips[:, 0], flips[:, 1]] = np.random.randint(2, size=noise)
    X = np.zeros((n_samples, size, size, 2))
    ix, iy, iz = np.ogrid[:X.shape[0], :X.shape[1], :X.shape[2]]
    X[ix, iy, iz, Y_flips] = 1
    X = X
    return X, Y


binary = [generate_blocks, generate_checker, generate_big_checker,
          generate_easy]

multinomial = [generate_blocks_multinomial, generate_checker_multinomial,
               generate_big_checker_extended, generate_easy_explicit]
