#!/usr/bin/env python

from netneurotools.networks import struct_consensus
import warnings
import numpy as np


# Setup hemisphere/distance info for structural consensus
hemi = np.array([0] * 41 + [1]*42)
hemi = hemi.reshape(-1, 1)
dists = np.loadtxt('../data/DK_res-1x1x1_distances.mat')

def structcon(gs):
    # Until the PR is merged on this package, this makes weighted consensus maps
    return np.mean(gs, axis=2)*struct_consensus(gs, dists, hemi)


def passthrough(gs):
    # Returns all the graphs as is; dummy function
    return gs


def refSample(gs, index=0):
    return gs[:,:,index]


def refTrunc(gs, index=0):
    # Compute edge-wise significant digits
    digits = sigdig(gs, base=10, axis=2)

    # Grab reference graph and non-zero locations
    ref = refSample(gs, index=index)
    loc = np.where(ref != 0)

    def trunc(value, prec):
        fstr = "{0:1." + str(np.max([0, prec-1])) + "e}"
        return np.float64(fstr.format(value)).astype(value.dtype)

    for l0, l1 in zip(loc[0], loc[1]):
        ref[l0, l1] = trunc(ref[l0, l1, 0], digits[l0, l1])

    return ref


def unstratifiedSample(gs, verbose=False):
    # Pick random value in a list and return the corresponding graph
    idx = np.random.choice(gs.shape[-1], size=1)[0]
    if verbose:
        print(idx)
    return gs[:,:,idx]


def sigdig(array, base=2, axis=-1):
    try:
        # If we have a float, this is our value of epsilon
        eps = np.finfo(array.dtype).eps
    except ValueError:
        # If it's an int, we want to convert it to a float of the same number of
        # bits to get our estimate of epsilon
        a2_dtype = "np.float{0}".format(array.dtype.itemsize*8)
        a2 = array.astype(eval(a2_dtype))
        warnings.warn("Converting array from {} to {}".format(array.dtype,
                                                              a2.dtype),
                      Warning)
        # Re-call the function with the float version
        return sigdig(a2, base=base, axis=axis)

    # Initialize empty matrix the same size of the array
    shp = list(array.shape)
    shp.pop(axis)
    sigs = np.empty(shp)
    sigs[:] = np.NaN

    # Compute the standard deviation and handle special case 1:
    #   - if no variance, maximum significance
    sd = np.std(array, axis=axis)
    c1locs = np.where(sd == 0)
    sigs[c1locs] = -np.log(eps)/np.log(base)

    # Compute the mean and handle special case 2:
    #   - if mean of 0, no significance.
    #   - N.B. this is the incorrect formula for zero-centered data
    mn = np.mean(array, axis=axis)
    c2locs = np.where(mn == 0)
    for c2l in zip(*c2locs):
        if np.isnan(sigs[c2l]):
            sigs[c2l] = 0

    # Otherwise, compute the number of significant digits using Parker, 1997
    c3locs = np.where(np.isnan(sigs))
    for c3l in zip(*c3locs):
        sigs[c3l] = -np.log(sd[c3l] / mn[c3l] + eps)/np.log(base)

    # Reset any negative values to zero
    c4l = np.where(sigs <= 0)
    sigs[c4l] = 0

    # Round up to nearest full bit, and return
    sigs = np.ceil(sigs).astype(int)
    return sigs


def qcAggregate(gs, agg, g_ind=0, log=True):
    import matplotlib.pyplot as plt
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


