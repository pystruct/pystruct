# these are some grid datasets to play with the algorithms.
# some are solable with submodular constraints, some are not.
# some need latent variables, some need directions.

import numpy as np


#### binary
def generate_blocks(n_samples=10, noise=1.5, seed=0):
    np.random.seed(seed)
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


def generate_big_checker(n_samples=20, noise=.5, n_fields=3, field_size=3):
    np.random.seed(0)
    y_small = np.ones((n_fields, n_fields), dtype=np.int32)
    y_small[::2, ::2] = -1
    y_small[1::2, 1::2] = -1
    y = y_small.repeat(field_size, axis=0).repeat(field_size, axis=1)
    Y = np.repeat(y[np.newaxis, :, :], n_samples, axis=0)
    X = Y + noise * np.random.normal(size=Y.shape)
    Y = (Y < 0).astype(np.int32)
    # make unaries
    X = np.r_['-1, 4,0', X, -X].copy("C")
    return X, Y


def make_simple_2x2(seed=0, n_flips=4, n_samples=20):
    np.random.seed(seed)
    X = []
    Y = []
    for i in range(n_samples):
        y = np.zeros((4, 4), dtype=np.int)
        j, k = 2 * np.random.randint(2, size=2)
        y[j: j + 2, k: k + 2] = 1
        Y.append(y)
        x = y.copy()
        for flip in range(n_flips):
            a, b = np.random.randint(4, size=2)
            x[a, b] = np.random.randint(2)
        x[x == 0] = -1
        X.append(x)
    return X, Y


def generate_easy(n_samples=5, noise=5, box_size=3, total_size=8, seed=0):
    np.random.seed(seed)
    Y = np.zeros((n_samples, total_size, total_size), dtype=np.int)
    for i in range(n_samples):
        t_old, l_old = -10, -10
        for j in range(2):
            #while True:
                #t, l = np.random.randint(1, size - 3, size=2)
                #if (t, l) in [(4, 4)]:
                    #continue
                #if np.abs(t - t_old) > 3 or np.abs(l - l_old) > 3:
                    #break
                #print(t, l, t_old, l_old)
            t, l = np.random.randint(1, total_size - box_size, size=2)
            if np.abs(t - t_old) > 3 or np.abs(l - l_old) > 3:
                t_old = t
                l_old = l
                Y[i, t:t + box_size, l:l + box_size] = 1

    Y_flips = Y.copy()
    for y in Y_flips:
        flips = np.random.randint(total_size, size=[noise, 2])
        y[flips[:, 0], flips[:, 1]] = np.random.randint(2, size=noise)
    X = np.zeros((n_samples, total_size, total_size, 2))
    ix, iy, iz = np.ogrid[:X.shape[0], :X.shape[1], :X.shape[2]]
    X[ix, iy, iz, Y_flips] = 1
    return X, Y


def generate_bars(n_samples=5, noise=5, bars_size=3, total_size=8,
                  random_seed=0, separate_labels=True):
    np.random.seed(random_seed)
    Y = np.zeros((n_samples, total_size, total_size), dtype=np.int)
    for i in range(n_samples):
        t_old, l_old = -10, -10
        for j in range(2):
            #while True:
                #t, l = np.random.randint(1, size - 3, size=2)
                #if (t, l) in [(4, 4)]:
                    #continue
                #if np.abs(t - t_old) > 3 or np.abs(l - l_old) > 3:
                    #break
                #print(t, l, t_old, l_old)
            t, l = np.random.randint(1, total_size - bars_size, size=2)
            if np.abs(t - t_old) > 3 or np.abs(l - l_old) > 3:
                t_old = t
                l_old = l
                #Y[i, t:t + box_size, l:l + box_size] = 1
                if np.random.uniform() > .5:
                    Y[i, t:t + bars_size, l] = 2 if separate_labels else 1
                else:
                    Y[i, t, l:l + bars_size] = 1

    Y_flips = Y.copy()
    n_classes = len(np.unique(Y))
    for y in Y_flips:
        flips = np.random.randint(total_size, size=[noise, 2])
        y[flips[:, 0], flips[:, 1]] = np.random.randint(n_classes, size=noise)
    X = np.zeros((n_samples, total_size, total_size, n_classes))
    ix, iy, iz = np.ogrid[:X.shape[0], :X.shape[1], :X.shape[2]]
    X[ix, iy, iz, Y_flips] = 1
    return X, Y


def generate_square_with_hole(n_samples=5, noise=5, total_size=8):
    box_size = 3
    np.random.seed(0)
    Y = np.zeros((n_samples, total_size, total_size), dtype=np.int)
    for i in range(n_samples):
        t_old, l_old = -10, -10
        t, l = np.random.randint(1, total_size - box_size, size=2)
        Y[i, t:t + box_size, l:l + box_size] = 1
        Y[i, t + 1, l + 1] = 0

    Y_flips = Y.copy()
    for y in Y_flips:
        flips = np.random.randint(total_size, size=[noise, 2])
        y[flips[:, 0], flips[:, 1]] = np.random.randint(2, size=noise)
    X = np.zeros((n_samples, total_size, total_size, 2))
    ix, iy, iz = np.ogrid[:X.shape[0], :X.shape[1], :X.shape[2]]
    X[ix, iy, iz, Y_flips] = 1
    return X, Y


def generate_crosses(n_samples=5, noise=30, total_size=10, n_crosses=2,
                     seed=0):
    np.random.seed(seed)
    Y = np.zeros((n_samples, total_size, total_size), dtype=np.int)
    for i in range(n_samples):
        t_old, l_old = -3, -3
        for j in range(n_crosses):
            while True:
                t, l = np.random.randint(1, total_size - 3, size=2)
                if np.abs(t - t_old) > 2 or np.abs(l - l_old) > 2:
                    break
            t_old = t
            l_old = l
            Y[i, t + 1, l:l + 3] = 1
            Y[i, t:t + 3, l + 1] = 1
            Y[i, t + 1, l + 1] = 1
    Y_flips = Y.copy()
    #flip random bits
    for y in Y_flips:
        flips = np.random.randint(1, total_size - 1, size=[noise, 2])
        y[flips[:, 0], flips[:, 1]] = np.random.randint(2, size=noise)
    X = np.zeros((n_samples, total_size, total_size, 2))
    ix, iy, iz = np.ogrid[:X.shape[0], :X.shape[1], :X.shape[2]]
    X[ix, iy, iz, Y_flips] = 1
    return X, Y


def generate_xs(n_samples=5, noise=30):
    np.random.seed(0)
    size = 8
    Y = np.zeros((n_samples, size, size), dtype=np.int)
    for i in range(n_samples):
        t_old, l_old = -3, -3
        for j in range(1):
            while True:
                t, l = np.random.randint(1, size - 3, size=2)
                if np.abs(t - t_old) > 2 and np.abs(l - l_old):
                    break
            t_old = t
            l_old = l
            Y[i, t + 1, l + 1] = 1
            Y[i, t + 2, l + 2] = 1
            Y[i, t + 3, l + 3] = 1
            Y[i, t + 1, l + 3] = 1
            Y[i, t + 3, l + 1] = 1
    Y_flips = Y.copy()
    #flip random bits
    for y in Y_flips:
        flips = np.random.randint(1, size - 1, size=[noise, 2])
        y[flips[:, 0], flips[:, 1]] = np.random.randint(2, size=noise)
    X = np.zeros((n_samples, size, size, 2))
    ix, iy, iz = np.ogrid[:X.shape[0], :X.shape[1], :X.shape[2]]
    X[ix, iy, iz, Y_flips] = 1
    return X, Y


#### Multinomial
def generate_blocks_multinomial(n_samples=20, noise=0.5, seed=None, size_x=12):
    if seed is not None:
        np.random.seed(seed)
    Y = np.zeros((n_samples, size_x - 2, size_x, 3))
    step = size_x // 3
    Y[:, :, :step, 0] = 1
    Y[:, :, step:-step, 1] = 1
    Y[:, :, -step:, 2] = 1
    X = Y + noise * np.random.normal(size=Y.shape)
    Y = np.argmax(Y, axis=3).astype(np.int32)
    return X, Y


def generate_checker_multinomial(n_samples=20, noise=1.5, size_x=12):
    Y = -np.ones((n_samples, size_x - 2, size_x, 3))
    Y[:, ::2, ::2, 0] = 1
    Y[:, 1::2, 1::2, 1] = 1
    Y[:, :, :, 2] = 0
    X = Y + noise * np.random.normal(size=Y.shape)
    Y = np.argmax(Y, axis=3).astype(np.int32)
    return X, Y


def generate_big_checker_extended(n_samples=20, noise=.3, n_fields=6,
                                  field_size=3):
    y_small = np.zeros((n_fields, n_fields), dtype=np.int32)
    y_small[::2, ::2] = 2
    y_small[1::2, 1::2] = 2
    y = y_small.repeat(field_size, axis=0).repeat(field_size, axis=1)
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


def generate_easy_explicit(n_samples=5, noise=5):
    size = 9
    np.random.seed(0)
    Y = np.zeros((n_samples, size, size), dtype=np.int)
    for i in range(n_samples):
        t_old, l_old = -4, -4
        for j in range(1):
            #while True:
                #t, l = np.random.randint(1, size - 3, size=2)
                #if (t, l) in [(4, 4)]:
                    #continue
                #if np.abs(t - t_old) > 3 or np.abs(l - l_old) > 3:
                    #break
                #print(t, l, t_old, l_old)
            t, l = np.random.randint(1, size - 3, size=2)
            if np.abs(t - t_old) > 3 or np.abs(l - l_old) > 3:
                t_old = t
                l_old = l
                Y[i, t:t + 3, l:l + 3] = 1
                Y[i, t + 1, l:l + 3] = 2
                Y[i, t:t + 3, l + 1] = 2
                Y[i, t + 1, l + 1] = 3

    Y_flips = (Y.copy() != 0).astype(np.int)
    for y in Y_flips:
        flips = np.random.randint(size, size=[noise, 2])
        y[flips[:, 0], flips[:, 1]] = np.random.randint(2, size=noise)
    X = np.zeros((n_samples, size, size, 4))
    ix, iy, iz = np.ogrid[:X.shape[0], :X.shape[1], :X.shape[2]]
    X[ix, iy, iz, Y_flips] = 1
    X[ix, iy, iz, Y_flips * 2] = 1
    X[ix, iy, iz, Y_flips * 3] = 1
    return X, Y


def generate_crosses_explicit(n_samples=5, noise=30, size=9, n_crosses=2):
    np.random.seed(0)
    Y = np.zeros((n_samples, size, size), dtype=np.int)
    for i in range(n_samples):
        t_old, l_old = -3, -3
        for j in range(n_crosses):
            while True:
                t, l = np.random.randint(1, size - 1, size=2)
                if np.abs(t - t_old) > 2 or np.abs(l - l_old) > 2:
                    break
            t_old = t
            l_old = l
            Y[i, t + 1, l:l + 3] = 1
            Y[i, t:t + 3, l + 1] = 1
            Y[i, t + 1, l + 1] = 2
    # don't distinguish between 2 and 3 in X
    Y_flips = (Y.copy() != 0).astype(np.int)
    #flip random bits
    for y in Y_flips:
        flips = np.random.randint(1, size - 1, size=[noise, 2])
        y[flips[:, 0], flips[:, 1]] = np.random.randint(2, size=noise)
    X = np.zeros((n_samples, size, size, 3))
    ix, iy, iz = np.ogrid[:X.shape[0], :X.shape[1], :X.shape[2]]
    X[ix, iy, iz, Y_flips] = 1
    X[ix, iy, iz, 2 * Y_flips] = 1
    return X, Y


def generate_crosses_latent(n_samples=5, noise=30):
    # X knows two states, Y knows four.
    np.random.seed(0)
    size = 8
    Y = np.zeros((n_samples, size, size), dtype=np.int)
    for i in range(n_samples):
        for j in range(2):
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

multinomial = [generate_blocks_multinomial, generate_checker_multinomial]
