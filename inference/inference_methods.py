import numpy as np
from pyqpbo import alpha_expansion_general_graph
from daimrf import mrf
import AD3

from .linear_programming import lp_general_graph

from IPython.core.debugger import Tracer
tracer = Tracer()


def inference_qpbo(x, unary_params, pairwise_params, edges):
    n_states = x.shape[-1]
    unaries = (-1000 * unary_params * x).copy().astype(np.int32)
    unaries = unaries.reshape(-1, n_states)
    pairwise = (-1000 * pairwise_params).copy().astype(np.int32)
    edges = edges.astype(np.int32)
    if pairwise_params.shape == (n_states, n_states):
        # only one matrix given
        edge_weights = np.repeat(pairwise[np.newaxis, :, :],
                                 edges.shape[0], axis=0)
    else:
        if pairwise_params.shape != (edges.shape[0], n_states, n_states):
            raise ValueError("Expected pairwise_params either to "
                             "be of shape n_states x n_states "
                             "or n_edges x n_states x n_states, but"
                             " got shape %s" % repr(pairwise_params.shape))
        edge_weights = pairwise
    y = alpha_expansion_general_graph(edges, unaries, edge_weights,
                                      random_seed=1)
    return y.reshape(x.shape[:-1])


def inference_dai(x, unary_params, pairwise_params, edges):
    ## build graph
    n_states = x.shape[-1]
    log_unaries = unary_params * x.reshape(-1, n_states)
    max_entry = max(np.max(log_unaries), 1)
    unaries = np.exp(log_unaries / max_entry)

    y = mrf(unaries, edges, np.exp(pairwise_params / max_entry), alg='jt')
    y = y.reshape(x.shape[:-1])

    return y


def inference_lp(x, unary_params, pairwise_params, edges, relaxed=False,
                 return_energy=False, exact=False):
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
    #if n_fractional:
        #print("got fractional solution. trying again, this time exactly")
        #res = lp_general_graph(-unaries, edges, -edge_weights, exact=True)
        #unary_marginals, pairwise_marginals, energy = res
        #n_fractional = np.sum(unary_marginals.max(axis=-1) < .9)
    if n_fractional:
        print("fractional solutions found: %d" % n_fractional)
    if relaxed:
        unary_marginals = unary_marginals.reshape(x.shape)
        if pairwise_params.shape == (n_states, n_states):
            pairwise_accumulated = pairwise_marginals.sum(axis=0)
            pairwise_accumulated = pairwise_accumulated.reshape(x.shape[-1],
                                                                x.shape[-1])
            y = (unary_marginals, pairwise_accumulated)
        else:
            y = (unary_marginals, pairwise_marginals)
    else:
        y = np.argmax(unary_marginals, axis=-1)
        y = y.reshape(x.shape[:-1])
    if return_energy:
        return y, energy
    return y


def inference_ad3(x, unary_params, pairwise_params, edges, relaxed=False,
                  verbose=0):
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
    res = AD3.general_graph(unaries, edges, edge_weights, verbose=verbose)
    unary_marginals, pairwise_marginals, energy = res
    n_fractional = np.sum(unary_marginals.max(axis=-1) < .99)
    if n_fractional:
        print("fractional solutions found: %d" % n_fractional)
    if relaxed:
        unary_marginals = unary_marginals.reshape(x.shape)
        if pairwise_params.shape == (n_states, n_states):
            pairwise_accumulated = pairwise_marginals.sum(axis=0)
            pairwise_accumulated = pairwise_accumulated.reshape(x.shape[-1],
                                                                x.shape[-1])
            y = (unary_marginals, pairwise_accumulated)
        else:
            y = (unary_marginals, pairwise_marginals)
    else:
        y = np.argmax(unary_marginals, axis=-1)
        y = y.reshape(x.shape[:-1])
    return y
