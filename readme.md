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

`|| y - Ax ||_2^2 + w * sum_i (log (|x_i| + e))`

where `w` is the weight constant and `e` is a very small constant (such as `1e-300`).

This program minimizes the objective function by coordinate descent.


