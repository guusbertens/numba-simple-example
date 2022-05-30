#!/usr/bin/env python3

import numba
import numpy as np
from mytimeit import mytimeit

N = 2**27  # 2^27 * 2^3 = 1 GB
A = np.random.rand(N)

def f_numpy(A, n):
    B = np.zeros_like(A)
    for j in range(n):
        B += A**j
    return B

def f_numpy2(A, n):
    '''Avoid exponentiation; replace with telescoping sum-product.'''
    B = np.zeros_like(A)
    for j in range(n):
        B = B * A + 1.0
    return B

def f_numpy3(A, n):
    '''Avoid memory allocation; reuse existing array.'''
    B = np.zeros_like(A)
    for j in range(n):
        B *= A
        B += 1.0
    return B

@numba.vectorize
def f_numba(A, n):
    '''Avoid RAM access and let Numba vectorize this.'''
    B = 0.0
    for j in range(n):
        B = B * A + 1.0
    return B

@numba.vectorize(['float64(float64,uint64)'], target='parallel')
def f_parallel(A, n):
    '''Let Numba parallelize this as well.'''
    B = 0.0
    for j in range(n):
        B = B * A + 1.0
    return B

mytimeit('f_numpy   (A, 4)', n=1)
mytimeit('f_numpy2  (A, 4)')
mytimeit('f_numpy3  (A, 4)')
mytimeit('f_numba   (A, 4)')
mytimeit('f_parallel(A, 4)')

# vim: ts=4 sw=4 et ai sta foldmethod=indent
