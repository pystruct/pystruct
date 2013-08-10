# cython: boundscheck=False
# cython: wraparound=False

def crammer_singer_psi(double[:,:] X, long[:] Y, double[:, :] out):
    cdef int y, i
    for i in xrange(X.shape[0]):
        y = Y[i]
        for j in xrange(X.shape[1]):
            out[y, j] += X[i, j]


def loss_augment_unaries(double[:,:] unary_potentials, long[:] y, double[:] class_weight):
    cdef int i
    cdef int n_states = unary_potentials.shape[1]
    for i in range(unary_potentials.shape[0]):
        for s in range(n_states):
            if s == y[i]:
                continue
            unary_potentials[i, s] += class_weight[s]
