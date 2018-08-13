---
layout: post
title: "Pydygp and Scikit-Learn"
---

## Introduction

In an attempt to avoid reinventing the wheel and to make use of their kernel library I am gradually phasing out my own implementations `Kernel` and `GaussianProcess` class in favour of the implementations in [`sklearn.gaussian_process`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process). This is relatively straightforward for the successive approximation method I am implementing, but the adaptive gradient matching methods make use gradient kernels, that is for a given kernel $$k(s, t)$$  we need implementations of

$$
\frac{\partial k(s, t)}{\partial t}, \qquad \frac{\partial^2 k(s, t)}{\partial s \partial t}.
$$

as well as the gradients of these new functions with respect to the kernel parameters. That is we need
1. First and second derivatives of kernels with respect to their arguments.
2. Gradients of these functions with respect to parameters.
3. Implementation of `__mul__`, `__add__` etc. for these kernels.

This blog post sketches out the process of doing 1. and 2. and serves as an invitation to anyone who would like to contribute.

## Gradient Kernel

First thing we are going to do is create a `GradientKernel` class which extends the basic functionality of `sklearn.gaussian_process.kernels.Kernel`.

```python
import sklearn.gaussian_process.kernels as sklearn_kernels

class GradientKernel(sklearn_kernels.Kernel):
    """
    Base class for the gradient kernel.
    """
    def __mul__(self, b):
        if isinstance(b, GradientKernel):
	    return GradientKernelProduct(self, b)
	else:
	    raise ValueError("Mulitplication must be between two GradientKernels")
```

So far so simple, all of the heavy lifting is still being done by the `Kernel` class in `sklearn`. But we need to tackle point 1. in the list above so we are going to override the behaviour of `__call__`. As an example lets consider the Radial Basis Function (RBF) kernel which is parameterised in `Kernels.RBF` as

$$
k_{RBF}(\mathbf{x}, \mathbf{y} ; \ell) = \exp\left\{ -\frac{1}{2}\sum_{i=1}^D \frac{(x_i - y_i)^2}{\ell_i^2}\right\}.
$$

Then

$$
\begin{align}
\frac{\partial k_{RBF}}{\partial y_j} = \frac{(x_j - y_j)}{\ell_j^2} k(\mathbf{x}, \mathbf{y} ; \ell).
\end{align}
$$

So lets try implementing this
```python
class RBF(GradientKernel, sklearn_kernels.RBF):

    def __call__(self, X, Y=None, eval_gradient=False, comp='x'):
        if comp == 'x':
	    return super(RBF, self).__call__(X, Y, eval_gradient)

        else:

            X = np.atleast_2d(X)
	    # see sklearn.gaussian_process.kernels for definition
	    # of this handler function
	    length_scale = _check_length_scale(X, self.length_scale)

            if Y is None:
	        Y = X

            # array of ( pairwise subtraction of X[:, j], Y[:, j], for j=1,...,D)
            Diffs = (X[:, np.newaxis, :] - Y[np.newaxis, :, :])
	    K = np.exp(-.5 * np.sum(Diffs ** 2 / (length_scale ** 2), axis=2)

	    if comp == 'xdx':

                Kdx = (Diffs / (length_scale ** 2) ) * K[..., np.newaxis]
		return Kdx

            else:
	        raise NotImplementedError
```
So now if `X` is a `(N, D)` array then `.__call__(X, comp='xdx')` is going to return a `(N, N, D)` array `Kdx` such that `Kdx[i, j, d]` is equal to

$$
\frac{\partial }{\partial y_d} k_{RBF}(\mathbf{x}, \mathbf{y} ; \ell)\bigg|_{\mathbf{x}=\mathbf{x}_i, \mathbf{y}=\mathbf{x}_j}.
$$

If we don't specify the component then the default behaviour is to implement the call method of the parent radial basis function class. Still do do then is implement the second derivative so that call can handle the argument `k(X, Y, comp=dxdx)` which will then return the `(N, M, D, D)` array. Where `N` and `M` correspond to the number of samples of `X` and `Y` respectively. A reasonably co

## Products of Gradient Kernels

Let us imagine we have sat down and done the work in the previous section for several kernels and now we would like to be able to freely transform these kernels to new kernels in such a way that the gradient kernel structure is still respected. That is for kernel functions \\(k_1, k_2\\) we would like to consider their product

$$
k_{prod} = k_1(x, y) \cdot k_2(x, y).
$$

Then as a gradient kernel we have

$$
\frac{\partial k_{prod}(\mathbf{x}, \mathbf{y})}{\partial y_d} = \frac{\partial k_1(\mathbf{x},\mathbf{y})}{\partial y_d} \cdot k_2(\mathbf{x}, \mathbf{y}) + k_1(\mathbf{x}, \mathbf{y})\frac{\partial k_2(\mathbf{x}, \mathbf{y})}{\partial y_d}
$$

or in terms of the code something like
```python
# Non conformable array shapes
Kprod_dx = k1(X, Y, comp='xdx') * k2(X, Y) + k1(X, Y) * k2(X, Y, comp='xdx')
```
as it stands we are trying to perform element wise multiplication of an `(N, M, D)` array with an `(N, M)` array, but this can be remedied by adding a new axis so that the array is now of shape `(N, M, 1)` and this will allow for `numpy`'s array broadcasting
```python
# This will work
Kprod_dx = k1(X, Y, comp='xdx') * k2(X, Y)[..., np.newaxis] + \
           k1(X, Y)[..., np.newaxis] * k2(X, Y, comp='xdx')
```
So now we just need to put this together inside a `GradientKernelProduct` class which we will make extend `sklearn.gaussian_process.kernels.Product` (which itself extends a more abstract `KernelOperator` class). Doing this we start to create something like
```python
class GradientKernelProduct(sklearn_kernels.Product):

    def __call__(self, X, Y=None, eval_gradient=False, comp='x'):
        if comp == 'x':
            return super(GradientKernelProduct, self).__call__(X, Y, eval_gradient=eval_gradient)

        elif comp == 'xdx':
	    if eval_gradient:
	        raise NotImplementedError
	    else:
	        K1 = self.k1(X, Y)
		K1dx = self.k1(X, Y, comp='xdx')
                K2 = self.k2(X, Y)
                K2dx = self.k2(X, Y, comp='xdx')
                return K1dx * K2[..., np.newaxis] + K1[..., np.newaxis] * K2dx
```
So now we are getting somewhere, we still need to add the method to handle the second derivatives and the gradients with respect to kernel parameters. Although note that the kernels are assumed to be distinct and therefore don't share parameters. As an example we could add the following inside the `if eval_gradient` block
```python
                K1, K1_gradient = self.k1(X, Y, eval_gradient=True)
                K1dx, K1dx_gradient = self.k1(X, Y, comp='xdx', eval_gradient=True)
                K2, K2_gradient = self.k2(X, Y, eval_gradient=True)
                K2dx, K2dx_gradient = self.k2(X, Y, comp='xdx', eval_gradient=True)

                # gradient wrt first par.
                grad1 = K1dx_gradient * K2[..., np.newaxis, np.newaxis] + \
                        K1_gradient[...,np.newaxis, :] * K2dx[..., np.newaxis]

                # gradient wrt second par.
                grad2 = K1dx[..., np.newaxis] * K2_gradient[..., np.newaxis, :] + \
                        K1[..., np.newaxis, np.newaxis] * K2dx_gradient

                Kdx = K1dx * K2[..., np.newaxis] + K1[..., np.newaxis] * K2dx
                Kdx_gradient = np.stack((grad1, grad2), axis=3)

                return Kdx, Kdx_gradient[...,0]
```
In this block we independent consider the gradient of the first kernel and of then second and then combine them through `np.stack((grad1, grad2), axis=3)`, this is now going to be an array of shape `(N, N, D, P)` where `P` is the sum of the free parameters of the two kernels.

## Onwards