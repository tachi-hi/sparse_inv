# Log-penalty-regularized sparse inverse

My exercise on Python/Numpy/PyUblas/Boost and sparse regularization.
Note that this program is far from optimal.

## Specification
### Problem setting

`y = Ax + noise` where `y` is an `n` dim vector, `A` is an `n x m` matrix, and `x` is an `m` dim vector where `m < n`.
`k` elements of `x` are nonzero, and other `m - k` elements are exactly 0.

Given `y` and `A` we estimate the unknown vector `x`.
(`k` and `noise`, and `x` of course, are unknown.)

### Strategy

This program solves the problem by minimizing following objective function.

`U = || y - Ax ||_2^2 + w * sum_i (log (x_i^2 + e))`

where `w` is the weight constant and `e` is a very small constant (such as `1e-300`).

This program minimizes the objective function by coordinate descent.
(By solving `\partial U / \partial x_i = 0` w.r.t. `x_i` we can derive an updating rule for each coordinate.)

## Usage
### Setup

One should install following

+ Python and libraries
    + `numpy`, `matplotlib`, `scikit-learn`, 
+ PyUblas https://pypi.python.org/pypi/PyUblas
    + Follow the instruction http://documen.tician.de/pyublas/installing.html
    + see also http://d.hatena.ne.jp/saket/20120411/1334147735
+ C++ compliler and Boost

### Run
Then type following,

    python run.py

Then two graphs will be displayed (compared with LASSO in `scikit-learn`)
+ sparseness-error plot
+ estimated `x`

## Some trivial facts (memo for me)

Note that log penalty `1 + p/2 log (x^2 + exp(-2/p) )` approximates `lp` norm quite finely when `p approx 0`

## Relevant Articles
+ G. Gasso and A. Rakotomamonjy and S. Canu "Recovering sparse signals with a certain family of non-convex penalties and DC programming" IEEE Trans Sig Proc, 57(12), pp. 4686-4698, 2009
+ R. Mazumder and J. Friedman and T. Hastie, "Sparse Net: Coordinage Descent with Non-Convex Penalties" 2009
