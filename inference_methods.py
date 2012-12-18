import numpy as np
from pyqpbo import alpha_expansion_graph
from daimrf import mrf
from lp_new import lp_general_graph
import AD3

from IPython.core.debugger import Tracer
tracer = Tracer()


def _make_grid_edges(x, neighborhood=4, return_lists=False):
    if neighborhood not in [4, 8]:
        raise ValueError("neighborhood can only be '4' or '8', got %s" %
                         repr(neighborhood))
    inds = np.arange(x.shape[0] * x.shape[1]).reshape(x.shape[:2])
    inds = inds.astype(np.int64)
    right = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
    down = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
    edges = [right, down]
    if neighborhood == 8:
        upright = np.c_[inds[1:, :-1].ravel(), inds[:-1, 1:].ravel()]
        downright = np.c_[inds[:-1, :-1].ravel(), inds[1:, 1:].ravel()]
        edges.extend([upright, downright])
    if return_lists:
        return edges
    return np.vstack(edges)


def _inference_qpbo(x, unary_params, pairwise_params, edges):
    unaries = (-1000 * unary_params * x).astype(np.int32)
    unaries = unaries.reshape(-1, x.shape[-1])
    pairwise = (-1000 * pairwise_params).astype(np.int32)
    edges = edges.astype(np.int32)
    y = alpha_expansion_graph(edges, unaries, pairwise, random_seed=1)
    return y.reshape(x.shape[:2])


def _inference_dai(x, unary_params, pairwise_params, edges):
    ## build graph
    n_states = x.shape[-1]
    log_unaries = unary_params * x.reshape(-1, n_states)
    max_entry = max(np.max(log_unaries), 1)
    unaries = np.exp(log_unaries / max_entry)

    y = mrf(unaries, edges, np.exp(pairwise_params / max_entry), alg='jt')
    y = y.reshape(x.shape[:2])

    return y


def _inference_lp(x, unary_params, pairwise_params, edges,
                  relaxed=False, return_energy=False, exact=False):
    n_states = x.shape[-1]
    unaries = unary_params * x.reshape(-1, n_states)
    if pairwise_params.shape == (n_states, n_states):
        # only one matrix given
        edge_weights = np.repeat(pairwise_params[np.newaxis, :, :],
                                 edges.shape[0], axis=0)
    else:
        if pairwise_params.shape != (edges.shape[0], n_states, n_states):
            raise ValueError("Expected pairwise_params either to "
                             "be of shape n_states x n_states "
                             "or n_edges x n_states x n_states, but"
                             " got shape %s" % repr(pairwise_params.shape))
        edge_weights = pairwise_params
    res = lp_general_graph(-unaries, edges, -edge_weights, exact=exact)
    unary_marginals, pairwise_marginals, energy = res
    n_fractional = np.sum(unary_marginals.max(axis=-1) < .99)
    if n_fractional:
        print("got fractional solution. trying again, this time exactly")
        res = lp_general_graph(-unaries, edges, -edge_weights, exact=True)
        unary_marginals, pairwise_marginals, energy = res
        n_fractional = np.sum(unary_marginals.max(axis=-1) < .9)
    if n_fractional:
        print("fractional solutions found: %d" % n_fractional)
    if relaxed:
        unary_marginals = unary_marginals.reshape(x.shape)
        #height, width = x.shape[:-1]
        #horz = pairwise_marginals[:(width - 1) * height]
        ##horz = horz.reshape(height, width - 1, n_states ** 2)
        #vert = pairwise_marginals[(width - 1) * height:]
        ##vert = vert.reshape(height - 1, width, n_states ** 2)
        #pairwise_accumulated = horz.sum(axis=0) + vert.sum(axis=0)

        #pairwise_accumulated = pairwise_marginals.sum(axis=0)
        #pairwise_accumulated = pairwise_accumulated.reshape(x.shape[-1],
                                                            #x.shape[-1])
        y = (unary_marginals, pairwise_marginals)
    else:
        y = np.argmax(unary_marginals, axis=-1)
        y = y.reshape(x.shape[0], x.shape[1])
    if return_energy:
        return y, energy
    return y


def _inference_ad3(x, unary_params, pairwise_params, edges,
                   relaxed=False, verbose=0):
    raise NotImplementedError("AD3 doesn't work on graphs yet!")
    res = AD3.simple_grid(unary_params * x, pairwise_params, verbose=verbose)
    unary_marginals, pairwise_marginals, energy = res
    n_fractional = np.sum(unary_marginals.max(axis=-1) < .99)
    if n_fractional:
        print("fractional solutions found: %d" % n_fractional)
    if relaxed:
        unary_marginals = unary_marginals.reshape(x.shape)
        pairwise_accumulated = pairwise_marginals.sum(axis=0)
        pairwise_accumulated = pairwise_accumulated.reshape(x.shape[-1],
                                                            x.shape[-1])
        y = (unary_marginals, pairwise_accumulated)
    else:
        y = np.argmax(unary_marginals, axis=-1)
        y = y.reshape(x.shape[0], x.shape[1])
    return y
