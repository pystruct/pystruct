import numpy as np
from pyqpbo import alpha_expansion_general_graph
from daimrf import mrf
import AD3

from .linear_programming import lp_general_graph

from IPython.core.debugger import Tracer
tracer = Tracer()


def inference_dispatch(unary_potentials, pairwise_potentials, edges,
                       inference_method, relaxed=False, return_energy=False,
                       exact=False):
    if inference_method == "qpbo":
        return inference_qpbo(unary_potentials, pairwise_potentials, edges)
    elif inference_method == "dai":
        return inference_dai(unary_potentials, pairwise_potentials, edges)
    elif inference_method == "lp":
        return inference_lp(unary_potentials, pairwise_potentials, edges,
                            relaxed, return_energy=return_energy, exact=exact)
    elif inference_method == "ad3":
        return inference_ad3(unary_potentials, pairwise_potentials, edges,
                             relaxed)
    else:
        raise ValueError("inference_method must be 'lp', 'ad3', 'qpbo' or"
                         " 'dai', got %s" % inference_method)


def _validate_params(unary_potentials, pairwise_params, edges):
    n_states = unary_potentials.shape[-1]
    if pairwise_params.shape == (n_states, n_states):
        # only one matrix given
        pairwise_potentials = np.repeat(pairwise_params[np.newaxis, :, :],
                                        edges.shape[0], axis=0)
    else:
        if pairwise_params.shape != (edges.shape[0], n_states, n_states):
            raise ValueError("Expected pairwise_params either to "
                             "be of shape n_states x n_states "
                             "or n_edges x n_states x n_states, but"
                             " got shape %s" % repr(pairwise_params.shape))
        pairwise_potentials = pairwise_params
    return n_states, pairwise_potentials


def inference_qpbo(unary_potentials, pairwise_potentials, edges):
    shape_org = unary_potentials.shape[:-1]
    n_states, pairwise_potentials = \
        _validate_params(unary_potentials, pairwise_potentials, edges)

    unary_potentials = (-1000 * unary_potentials).copy().astype(np.int32)
    unary_potentials = unary_potentials.reshape(-1, n_states)
    pairwise_potentials = (-1000 * pairwise_potentials).copy().astype(np.int32)
    edges = edges.astype(np.int32)
    y = alpha_expansion_general_graph(edges, unary_potentials,
                                      pairwise_potentials, random_seed=1)
    return y.reshape(shape_org)


def inference_dai(unary_potentials, pairwise_potentials, edges):
    shape_org = unary_potentials.shape[:-1]
    n_states, pairwise_potentials = \
        _validate_params(unary_potentials, pairwise_potentials, edges)

    n_states = unary_potentials.shape[-1]
    log_unaries = unary_potentials.reshape(-1, n_states)
    max_entry = max(np.max(log_unaries), 1)
    unaries = np.exp(log_unaries / max_entry)

    y = mrf(unaries, edges, np.exp(pairwise_potentials / max_entry), alg='jt')
    y = y.reshape(shape_org)
    return y


def inference_lp(unary_potentials, pairwise_potentials, edges, relaxed=False,
                 return_energy=False, exact=False):
    shape_org = unary_potentials.shape[:-1]
    n_states, pairwise_potentials = \
        _validate_params(unary_potentials, pairwise_potentials, edges)

    unaries = unary_potentials.reshape(-1, n_states)
    res = lp_general_graph(-unaries, edges, -pairwise_potentials, exact=exact)
    unary_marginals, pairwise_marginals, energy = res
    n_fractional = np.sum(unary_marginals.max(axis=-1) < .99)
    if n_fractional:
        print("fractional solutions found: %d" % n_fractional)
    if relaxed:
        unary_marginals = unary_marginals.reshape(unary_potentials.shape)
        y = (unary_marginals, pairwise_marginals)
    else:
        y = np.argmax(unary_marginals, axis=-1)
        y = y.reshape(shape_org)
    if return_energy:
        return y, energy
    return y


def inference_ad3(unary_potentials, pairwise_potentials, edges, relaxed=False,
                  verbose=0):
    shape_org = unary_potentials.shape[:-1]
    n_states, pairwise_potentials = \
        _validate_params(unary_potentials, pairwise_potentials, edges)

    unaries = unary_potentials.reshape(-1, n_states)
    res = AD3.general_graph(unaries, edges, pairwise_potentials,
                            verbose=verbose)
    unary_marginals, pairwise_marginals, energy = res
    n_fractional = np.sum(unary_marginals.max(axis=-1) < .99)
    if n_fractional:
        print("fractional solutions found: %d" % n_fractional)
    if relaxed:
        unary_marginals = unary_marginals.reshape(unary_potentials.shape)
        y = (unary_marginals, pairwise_marginals)
    else:
        y = np.argmax(unary_marginals, axis=-1)
        y = y.reshape(shape_org)
    return y
