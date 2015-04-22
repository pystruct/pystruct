#!/usr/bin/python
"""
This module provides a callable for easy evaluation of stored models.
"""
import sys

import numpy as np

from pystruct.utils import SaveLogger


def main():
    argv = sys.argv
    print("loading %s ..." % argv[1])
    ssvm = SaveLogger(file_name=argv[1]).load()
    plot_learning(ssvm)


def plot_learning(ssvm, time=True):
    """Plot optimization curves and cache hits.

    Create a plot summarizing the optimization / learning process of an SSVM.
    It plots the primal and cutting plane objective (if applicable) and also
    the target loss on the training set against training time.
    For one-slack SSVMs with constraint caching, cached constraints are also
    contrasted against inference runs.

    Parameters
    -----------
    ssvm : object
        SSVM learner to evaluate. Should work with all learners.

    time : boolean, default=True
        Whether to use wall clock time instead of iterations as the x-axis.

    Notes
    -----
    Warm-starting a model might mess up the alignment of the curves.
    So if you warm-started a model, please don't count on proper alignment
    of time, cache hits and objective.
    """
    import matplotlib.pyplot as plt
    print(ssvm)
    if hasattr(ssvm, 'base_ssvm'):
        ssvm = ssvm.base_ssvm
    print("Iterations: %d" % len(ssvm.objective_curve_))
    print("Objective: %f" % ssvm.objective_curve_[-1])
    inference_run = None
    if hasattr(ssvm, 'cached_constraint_'):
        inference_run = ~np.array(ssvm.cached_constraint_)
        print("Gap: %f" %
              (np.array(ssvm.primal_objective_curve_)[inference_run][-1] -
               ssvm.objective_curve_[-1]))
    if hasattr(ssvm, "loss_curve_"):
        n_plots = 2
        fig, axes = plt.subplots(1, 2)
    else:
        n_plots = 1
        fig, axes = plt.subplots(1, 1)
        axes = [axes]
    if time and hasattr(ssvm, 'timestamps_'):
        print("loading timestamps")
        inds = np.array(ssvm.timestamps_)
        inds = inds[2:len(ssvm.objective_curve_) + 1] / 60.
        inds = np.hstack([inds, [inds[-1]]])
        axes[0].set_xlabel('training time (min)')
    else:
        inds = np.arange(len(ssvm.objective_curve_))
        axes[0].set_xlabel('QP iterations')

    axes[0].set_title("Objective")
    axes[0].plot(inds, ssvm.objective_curve_, label="dual")
    axes[0].set_yscale('log')
    if hasattr(ssvm, "primal_objective_curve_"):
        axes[0].plot(inds, ssvm.primal_objective_curve_,
                     label="cached primal" if inference_run is not None
                     else "primal")
    if inference_run is not None:
        inference_run = inference_run[:len(ssvm.objective_curve_)]
        axes[0].plot(inds[inference_run],
                     np.array(ssvm.primal_objective_curve_)[inference_run],
                     'o', label="primal")
    axes[0].legend()
    if n_plots == 2:
        if time and hasattr(ssvm, "timestamps_"):
            axes[1].set_xlabel('training time (min)')
        else:
            axes[1].set_xlabel('QP iterations')

        try:
            axes[1].plot(inds[::ssvm.show_loss_every], ssvm.loss_curve_)
        except:
            axes[1].plot(ssvm.loss_curve_)

        axes[1].set_title("Training Error")
        axes[1].set_yscale('log')
    plt.show()


if __name__ == "__main__":
    main()
