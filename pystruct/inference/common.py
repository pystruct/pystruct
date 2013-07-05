import numpy as np


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


def compute_energy(unary_potentials, pairwise_potentials, edges, labels):
    """Compute energy of labels for given energy function.

    Convenience function with same interface as inference functions to easily
    compare solutions.

    Parameters
    ----------
    unary_potentials : nd-array
        Unary potentials of energy function.

    pairwise_potentials : nd-array
        Pairwise potentials of energy function.

    edges : nd-array
        Edges of energy function.

    labels : nd-array
        Variable assignment to evaluate.

    Returns
    -------
    energy : float
        Energy of assignment.
    """

    n_states, pairwise_potentials = \
        _validate_params(unary_potentials, pairwise_potentials, edges)
    energy = np.sum(unary_potentials[np.arange(len(labels)), labels])
    for edge, pw in zip(edges, pairwise_potentials):
        energy += pw[labels[edge[0]], labels[edge[1]]]
    return energy
