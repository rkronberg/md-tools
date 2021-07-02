import numpy as np


def mean(list_of_arrays):
    '''Computes mean over arrays of different length.

    Args:
        list_of_arrays: List of arrays.

    Returns:
        Mean over the last axis.
    '''

    lengths = [len(a) for a in list_of_arrays]
    arr = np.ma.empty((np.max(lengths), len(list_of_arrays)))
    arr.mask = True
    for idx, l in enumerate(list_of_arrays):
        arr[:len(l), idx] = l

    return np.nanmean(arr, axis=-1)
