"""Utility functions"""
import numpy as np


def pad_rect(lists):
    """Turn a list of lists of arrays into a rectangular numpy array by padding with nans."""
    max_len = max(len(l) for l in lists)
    return np.array([l + [np.ones_like(l[0]) * np.nan] * (max_len - len(l)) for l in lists])
