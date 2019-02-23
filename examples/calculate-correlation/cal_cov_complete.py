from __future__ import division, absolute_import, print_function

try:
    # Accessing collections abstact classes from collections
    # has been deprecated since Python 3.3
    import collections.abc as collections_abc
except ImportError:
    import collections as collections_abc
import re
import sys
import warnings
import operator

import numpy as np
import numpy.core.numeric as _nx
from numpy.core import linspace, atleast_1d, atleast_2d, transpose
from numpy.core.numeric import (
    ones, zeros, arange, concatenate, array, asarray, asanyarray, empty,
    empty_like, ndarray, around, floor, ceil, take, dot, where, intp,
    integer, isscalar, absolute, AxisError
    )
from numpy.core.umath import (
    pi, multiply, add, arctan2, frompyfunc, cos, less_equal, sqrt, sin,
    mod, exp, log10, not_equal, subtract
    )
from numpy.core.fromnumeric import (
    ravel, nonzero, sort, partition, mean, any, sum
    )
from numpy.core.numerictypes import typecodes, number
from numpy.lib.twodim_base import diag
from numpy.lib.utils import deprecate
from numpy.core.multiarray import (
    _insert, add_docstring, digitize, bincount, normalize_axis_index,
    interp as compiled_interp, interp_complex as compiled_interp_complex
    )
from numpy.core.umath import _add_newdoc_ufunc as add_newdoc_ufunc
from numpy.compat import long
from numpy.compat.py3k import basestring
from numpy import average

if sys.version_info[0] < 3:
    # Force range to be a generator, for np.delete's usage.
    range = xrange
    import __builtin__ as builtins
else:
    import builtins

# needed in this module for compatibility
from numpy.lib.histograms import histogram, histogramdd

def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None,
aweights=None):
    # Check inputs
    if ddof is not None and ddof != int(ddof):
        raise ValueError(
            "ddof must be integer")
    # Handles complex arrays too
    m = np.asarray(m)
    if m.ndim > 2:
        raise ValueError("m has more than 2 dimensions")

    if y is None:
        dtype = np.result_type(m, np.float64)
    else:
        y = np.asarray(y)
        if y.ndim > 2:
            raise ValueError("y has more than 2 dimensions")
        dtype = np.result_type(m, y, np.float64)

    X = array(m, ndmin=2, dtype=dtype)
    if not rowvar and X.shape[0] != 1:
        X = X.T
    if X.shape[0] == 0:
        return np.array([]).reshape(0, 0)
    if y is not None:
        y = array(y, copy=False, ndmin=2, dtype=dtype)
        if not rowvar and y.shape[0] != 1:
            y = y.T
        X = np.concatenate((X, y), axis=0)

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    # Get the product of frequencies and weights
    w = None
    if fweights is not None:
        fweights = np.asarray(fweights, dtype=float)
        if not np.all(fweights == np.around(fweights)):
            raise TypeError(
                "fweights must be integer")
        if fweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional fweights")
        if fweights.shape[0] != X.shape[1]:
            raise RuntimeError(
                "incompatible numbers of samples and fweights")
        if any(fweights < 0):
            raise ValueError(
                "fweights cannot be negative")
        w = fweights
    if aweights is not None:
        aweights = np.asarray(aweights, dtype=float)
        if aweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional aweights")
        if aweights.shape[0] != X.shape[1]:
            raise RuntimeError(
                "incompatible numbers of samples and aweights")
        if any(aweights < 0):
            raise ValueError(
                "aweights cannot be negative")
        if w is None:
            w = aweights
        else:
            w *= aweights

    avg, w_sum = average(X, axis=1, weights=w, returned=True)
    w_sum = w_sum[0]

    # Determine the normalization
    if w is None:
        fact = X.shape[1] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof*sum(w*aweights)/w_sum

    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice",
                      RuntimeWarning, stacklevel=2)
        fact = 0.0

    X -= avg[:, None]
    if w is None:
        X_T = X.T
    else:
        X_T = (X*w).T
    c = dot(X, X_T.conj())
    c *= np.true_divide(1, fact)
    return c.squeeze()

if __name__ == '__main__':
    m = [0,1,2]
    print (cov(m))
