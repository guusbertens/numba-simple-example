#!/usr/bin/env python3

import numba
import numpy as np
import scipy.sparse
import sys
from mytimeit import mytimeit

# Prepare irregularly shaped "matrix"
Ntot = 2**27  # Total size [elem], 2^27 * 2^3 byte = 1 GB
Mmin = 100  # Minimum row length [elem]
Mavg = 10000  # Average row length [elem]
N = Ntot // Mavg  # Number of rows
M = (Mmin + 2 * (Mavg - Mmin) * np.random.rand(N)).astype(int)  # Number of columns per row
jind = np.insert(np.cumsum(M), 0, 0)
M = np.amax(M)
Ntot = jind[-1]

# Generate some data
X = np.random.rand(Ntot)

# Generate regular NumPy array as well.
Xnp = np.zeros((N, M))
for i in range(N): Xnp[i,:jind[i+1]-jind[i]] = X[jind[i]:jind[i+1]]

print(f'We have {N} rows')

def sum_python(X, jind):
    N = jind.shape[0] - 1
    s = 0
    for i in range(N):
        for j in range(jind[i], jind[i+1]):
            s += np.arctan(X[j] * i * j)
    return s

def sum_numpy(Xnp):
    i = np.arange(Xnp.shape[0])[:,None]
    j = np.arange(Xnp.shape[1])[None,:]
    s = np.sum(np.arctan(Xnp * i * j))
    return s

def sum_numpy2(X, jind):
    N = jind.shape[0] - 1
    s = 0
    for i in range(N):
        j = np.arange(jind[i+1] - jind[i])
        s += np.sum(np.arctan(X[jind[i]:jind[i+1]] * i * j))
    return s

@numba.njit
def sum_numba(X, jind):
    N = jind.shape[0] - 1
    s = 0
    for i in range(N):
        for j in range(jind[i], jind[i+1]):
            s += np.arctan(X[j] * i * j)
    return s

@numba.njit(parallel=True)
def sum_parallel(X, jind):
    N = jind.shape[0] - 1
    s = 0
    for i in numba.prange(N):
        for j in range(jind[i], jind[i+1]):
            s += np.arctan(X[j] * i * j)
    return s

mytimeit('sum_python  (X, jind[:100] )')
mytimeit('sum_numpy   (Xnp    [:4000])')
mytimeit('sum_numpy2  (X, jind       )')
mytimeit('sum_numba   (X, jind       )')
mytimeit('sum_parallel(X, jind       )')

# vim: ts=4 sw=4 et ai sta foldmethod=indent
