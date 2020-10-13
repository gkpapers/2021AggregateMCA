#!/usr/bin/env python

from netneurotools.networks import struct_consensus
import matplotlib.pyplot as plt
import numpy as np


# Setup hemisphere/distance info for structural consensus
hemi = np.array([0] * 41 + [1]*42)
hemi = hemi.reshape(-1, 1)
dists = np.loadtxt('../data/DK_res-1x1x1_distances.mat')

def structcon(gs):
    # Until the PR is merged on this package, this makes weighted consensus maps
    return np.mean(gs, axis=2)*struct_consensus(gs, dists, hemi)


def qcAggregate(gs, agg, g_ind=0, log=True):
    # Function to visualize example graph next to aggregate
    print("Original shape: ", gs.shape)
    print("Aggregation shape: ", agg.shape)

    func = lambda x: np.log10(x+1) if log else x

    fig = plt.Figure(figsize=(10, 5))
    plt.subplot(121)
    g = gs[:,:, g_ind]
    plt.imshow(func(g))

    plt.subplot(122)
    if len(agg.shape) == 3:
        agg = agg[:,:, g_ind]
    plt.imshow(func(agg))
    plt.show()
