# cython: boundscheck=False
# cython: wraparound=False

def crammer_singer_psi(double[:,:] X, long[:] Y, double[:, :] out):
    cdef int y, i
    for i in xrange(X.shape[0]):
        y = Y[i]
        for j in xrange(X.shape[1]):
            out[y, j] += X[i, j]
