# Numba: simple example & benchmarks

> Disclaimer: the benchmarks shown here are not meant to exhaustively test Numba; rather they are examples of how Numba can speed up numerical operations on NumPy arrays.

## What is Numba?

It is

* a just-in-time (JIT) compiler for Python
* that keeps your code simple
* meant for numerical operations
    * supports NumPy and multi-threading

It is not

* general-purpose
    * use Cython for that
    
## Preparation

```bash=
module load anaconda3
eval "$(conda shell.$(basename $SHELL) hook)"
conda create -n myenv numpy numba
conda activate myenv
```

## Compiling Python code with `@jit`

* `@jit` compiles a Python function
* `@njit` too, but disallows calling back into non-compiled Python code
* Regular Python: don't write loops!
    * If possible, use NumPy functions instead
* Numba: loops are OK
    * They're compiled anyway
    * In fact, Numba can parallelise them
* Some use cases:
    * Data and/or operations not easily expressed in arrays
    * Elementwise operations that slightly differ per element

### Example

Given a sequence of subsequences of various lengths, with elements $X_{ij}$, compute

$$ \sum_i \sum_j \arctan(ijX_{ij}) $$

We could treat $X$ as a 2D array, in which case it would have rows of different length, e.g.

$$X = \left(\begin{array}{ccccc}
1 & 3 & 4 & 5 & - \\
1 & - & - & - & - \\
1 & 3 & - & - & - \\
1 & 3 & 9 & 0 & 2 \\
\end{array}\right)$$

and store it as a normal array (NumPy) or as a sparse array (SciPy).  This makes for slightly convoluted expressions, but can be quite fast, still.

With Numba the expressions stay close to what is shown above, and it is even faster.  To access the elements of $X$, concatenate all subsequences, and construct an indexing array:

$$X = (
\color{red}   {1\ 3\ 4\ 5\ }
\color{green} {1\ }
\color{blue}  {1\ 3\ }
\color{purple}{1\ 3\ 9\ 0\ 2}
)$$

$$j_\text{ind} = (
\color{red}{0}\ 
\color{green}{4}\ 
\color{blue}{5}\ 
\color{purple}{7}\ 
12
)$$

Each element of $j_\text{ind}$ points to the start of a subsequence in $X$.  The last element of $j_\text{ind}$ points to one element beyond the end of $X$.

```Python
import numba
import numpy as np
import scipy.sparse
import sys
from mytimeit import mytimeit

# Prepare irregularly shaped "matrix";
# This is our sequence of subsequences.
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
```

Output:

```
We have 13421 rows
sum_python  (X, jind[:100] ): quickest out of 5 took 2.220273009967059 s
sum_numpy   (Xnp    [:4000]): quickest out of 5 took 2.304305628989823 s
sum_numpy2  (X, jind       ): quickest out of 5 took 3.0203089740825817 s
sum_numba   (X, jind       ): quickest out of 5 took 2.5618122160667554 s
sum_parallel(X, jind       ): quickest out of 5 took 0.15291113802231848 s
```

Note that `sum_python` and `sum_numpy` did only about 1% and 25% of the total work, respectively.

### Real world use case

Maximise the log-likelihood function $\ell$, given by

$$\ell = \log L$$

$$L = \prod_i \prod_j P( I_{ij} | \mu_{ij}, \sigma_{ij} ) $$

$$\mu_{ij} = \mu_i \cdot I_{0ij}$$

$$\sigma_{ij} = c_0 \cdot \mu_{ij}^{c_1}$$

with free variables $\mu_i$, $c_0$, and $c_1$, and $P$ the normal distribution.  $I_{ij}$ and $I_{0ij}$ are prescribed sequences of subsequences.

```Python
@njit
def ell(X, Nt, jind, I, I0):
    '''
    X[0:Nt] -> mu_i
    X[Nt:Nt+2] -> c0, c1
    i -> 0..Nt-1
    '''
    mu = X[0:Nt]
    c0, c1 = X[Nt:Nt+2]
    ell = 0
    for i in nb.prange(Nt):
        for j in range(jind[i], jind[i+1]):
            n = I[j] - mu[i] * I0[j]
            s = c0 * (mu[i] * I0[j])**c1
            ell += np.log(s) + 0.5 * n**2 / s**2
    return ell
```

The same strategy was applied to compute the gradient and Hessian of $\ell$.  Such derivates are sufficiently complicated as it is; besides the speed, Numba also helps to avoid writing complicated indexing expressions, which would further compilcate the code.

Tip: when coding ugly derivatives, *always* compare them with numerical derivatives (e.g. finite differences) for small inputs.

## Ufuncs with `@vectorize`

* NumPy universal functions
    * Operate on arrays
    * Operate elementwise
    * Examples: `np.arctan`, `np.add`
    * NumPy broadcasting for arrays of different sizes
* With Numba ...
    * Write an elementwise function
    * NumPy broadcasting still applies!
* Numba can run it in parallel as well

### Example

Compute

$$1 + A + A^2 + ... + A^{n-1}$$

for some $n$.  Note that this can be written as a telescopic expression as well:

$$1 + A \cdot (1 + A \cdot (1 + A \cdot (\ldots)))$$

```Python
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
```

Output:

```
f_numpy   (A, 4): quickest out of 1 took 15.506015082006343 s
f_numpy2  (A, 4): quickest out of 5 took 5.065712174982764 s
f_numpy3  (A, 4): quickest out of 5 took 2.4017412570538 s
f_numba   (A, 4): quickest out of 5 took 1.057333413977176 s
f_parallel(A, 4): quickest out of 5 took 0.21806608501356095 s
```

### Real world use case

Evaluate $\log P$ of a large number of slightly different normal distributions.

```Python
@numba.vectorize
def log_normal_pdf(x, mu, sigma):
    '''
    Returns ln(P(x|mu,sigma)).  The factor 1/sqrt(2pi) is omitted.
    '''
    return -0.5 * ((x - mu) / sigma)**2 - np.log(sigma)
```

This can be called with e.g. `x` a rectangular array, `mu` a row vector, and `sigma` a column vector.  NumPy's broadcasting rules will apply, which results in many fewer RAM accesses, which makes this run faster than the NumPy equivalent.

## Summary

In short:

* Numba brings JIT to Python, aimed at numerical tasks
* Often, only minimal changes to your own code are needed
* Speed increase can be very large

Not discussed:

* `@stencil` for applying kernels to arrays
* `@cuda.jit` for GPU computing

## Appendix: `mytimeit`

This code should be stored in a file `mytimeit.py`; it is used by the example codes, and assumes that they are called like `python3 example1.py` (or similar filename), so that they correspond to the `__main__` module.

Note: `import __main__` is *not* good programming, but is sufficient for small examples codes such as shown here.

```Python
import __main__
import timeit

def mytimeit(stmt, n=5):
    min_elapsed = min([
        timeit.timeit(stmt=stmt, number=1, globals=__main__.__dict__)
        for i in range(n)])
    print(f'{stmt}: quickest out of {n} took {min_elapsed} s')
```
