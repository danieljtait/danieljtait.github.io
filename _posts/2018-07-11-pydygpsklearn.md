---
layout: post
title: "Pydygp and Scikit-Learn Kernels"
---

## Introduction

In an attempt to avoid reinventing the wheel and to make use of and existing mature kernel library I am gradually phasing out my own implementations of `Kernel` and `GaussianProcess` class in favour of the implementations in [`sklearn.gaussian_process`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process). Not only will this hopefully mean the API is familiar to new users, but also because the kernel classes in scikit learn have a nicely implemented algebra - we can add and multiply kernels to create new kernels in a user friendly way to make complicated kernels out of simpler building blocks, and importantly the methods keep track of kernel hyperparameters and the gradient of the kernel with respect to these hyperparameters.

One challenge however for implementing these kernels in PydyGp is the use of the gradient Gaussian processes in the adaptive gradient matching methods. It is a wonderful feature of a Gaussian process $f(\mathbf{x})$ that the gradient processes $\\{ \frac{\partial }{\partial x_i} f(\mathbf{x}) \\}$ are themselves Gaussian processes, the joint distribution given by

$$
\operatorname{Cov}\left\{f(\mathbf{x}), \frac{\partial f(\mathbf{y})}{\partial y_p } \right\} = \frac{\partial k(\mathbf{x}, \mathbf{y})}{\partial y_p }, \qquad \operatorname{Cov}\left\{\frac{\partial f(\mathbf{x})}{\partial x_p}, \frac{\partial f(\mathbf{y})}{\partial y_q } \right\} = \frac{\partial^2 k(\mathbf{x}, \mathbf{y})}{\partial x_p \partial y_q },
$$

for any points $\mathbf{x}, \mathbf{y}$ in the input space. So if we are going to use the kernels in Scikit-Learn we are going to have to implement these gradients as well as the gradients of these new functions with respect to the parameters. That is we are going to need

1. First and second derivatives of kernels with respect to their arguments.
2. Gradients of these functions with respect to parameters.
3. Implementation of `__mul__`, `__add__` etc. so we can make an algebra of these kernels

This post sketches out some of the necessary details for implementing these gradient kernels, and hopefully a useful starting point for anyone who would like to contribute.

## A Gradient Kernel class

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
	    raise ValueError("Multiplication must be between two GradientKernels")
```

So far so simple - all of the heavy lifting is still being done by the `Kernel` class in `sklearn`. We have also begun the process of definining the multiplication between two gradient kernels objects although we still need to create the `GradientKernelProduct` class which will take two kernels and create something with the same basic functionality as a kernel. We should also probably overwrite the `__add__` method and so on otherwise this class will return nonsense, but that is left to the reader!

So to turn this into something useful we are going to need to override the behaviour of `__call__`. As an example lets consider the Radial Basis Function (RBF) kernel which is parameterised in `Kernels.RBF` as

$$
k_{RBF}(\mathbf{x}, \mathbf{y} ; \ell) = \exp\left\{ -\frac{1}{2}\sum_{i=1}^D \frac{(x_i - y_i)^2}{\ell_i^2}\right\}.
$$

Then, and watching for any mistakes I may have made to keep the reader on their toes, we can calculate the gradients

$$
\begin{align}
\frac{\partial k_{RBF}}{\partial y_j} = \frac{(x_j - y_j)}{\ell_j^2} k(\mathbf{x}, \mathbf{y} ; \ell).
\end{align}
$$

So lets try implementing this, and implementing this in a way that respects the broadcasting in `numpy`
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
	    K = np.exp(-.5 * np.sum(Diffs ** 2 / (length_scale ** 2), axis=2))

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

If we don't specify the component then the default behaviour is to ignore our additions and to implement the call method of the parent radial basis function class. Still to do then is to implement the second derivative so that call can handle the argument `k(X, Y, comp=dxdx)` which will then return the `(N, M, D, D)` Hessian array. Where `N` and `M` correspond to the number of samples of `X` and `Y` respectively, by symmetry of course there is a redundancy in returning the full Hessian so there is the option to make some small gains in speed and storage by doing this in smarter way but this is not something I have pursued - memory hasn't been a make or break factor and the computational bottleneck is usually the inversion of the full convariance matrix rather than the construction of that matrix. 

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
So now we just need to put this together inside a `GradientKernelProduct` class which shall extend the <a href="http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Product.html">`kernels.Product`</a> of scikit-learn (which itself extends a more abstract <a href="">`KernelOperator`</a> class). Doing this we start to create something like
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

	elif comp == 'dxdx':
	    # returns the cov{ dfdxp, dfdxq }
	    raise NotImplementedError("view this as an invitation")
```
So now we are getting somewhere, we still need to add the method to handle the second derivatives and the gradients with respect to kernel parameters, but by extending the base class we gain access to member functions of the parent `KernelProduct` class including utility methods for handling hyperparameters of the consitutent kernels of the product as well as methods to return flattened arrays of the (usually log transformed) hyperparameters. Returning gradients of the kernels with respect to hyperparameters is made easier because the kernels are assumed to be distinct, as an example we could add the following inside the `if eval_gradient` block
```python
class GradientKernelProduct(sklearn_kernels.Product):

    def __call__(self, X, Y=None, eval_gradient=False, comp='x'):
        if comp == 'x':
	    ...
      
        elif comp == 'xdx':
	    if eval_gradient:
	        raise NotImplementedError
                K1, K1_gradient = self.k1(X, Y, eval_gradient=True)
                K1dx, K1dx_gradient = self.k1(X, Y, comp='xdx', eval_gradient=True)
                K2, K2_gradient = self.k2(X, Y, eval_gradient=True)
                K2dx, K2dx_gradient = self.k2(X, Y, comp='xdx', eval_gradient=True)

                # gradient wrt first kernel's par
                grad1 = K1dx_gradient * K2[..., np.newaxis, np.newaxis] + \
                        K1_gradient[...,np.newaxis, :] * K2dx[..., np.newaxis]

                # gradient wrt second kernel's par
                grad2 = K1dx[..., np.newaxis] * K2_gradient[..., np.newaxis, :] + \
                        K1[..., np.newaxis, np.newaxis] * K2dx_gradient

                Kdx = K1dx * K2[..., np.newaxis] + K1[..., np.newaxis] * K2dx
                Kdx_gradient = np.stack((grad1, grad2), axis=3)

                return Kdx, Kdx_gradient[...,0]
```
In this block we independently consider the gradient of the first kernel and of then second and then combine them through `np.stack((grad1, grad2), axis=3)`, this is now going to be an array of shape `(N, N, D, P)` where `P` is the sum of the free parameters of the two kernels.

## Onwards

To really get this up and running then we still need to implement the methods for the second order derivatives etc. but hopefully the above makes it pretty clear how we would go about doing that. For my application it is  enough for me to just use the kernels as they are rather than build a Gaussian process class that accepts such a kernel as an input, but an implementation of just such a Gaussian process is still something I would like to implement and so the question becomes what should the API look like? One way would be to package everything up in such a way that you could still pass everything to the standard Gaussian process class and then just use that in the usual way, but in my opinion a dedicated `MultioutputGaussianProcess` class would be a more user friendly option. Something like
```python

kern = RBF()
gp = GradientGaussianProcess(kern)

gp.fit(X=...,     # training inputs for f(x)
       dX=...,    # training inputs (if any) for dfdx
       y=...,     # training obs for f(x)
       dydx=...)  # training obs for dfdx
```
what I am trying to convey is the possibility that we may want to fit/predict the Gaussian process using only gradient observations and so on. The `GradientGaussianProcess` would extend the more general `MultioutputGaussianProcess` which would probably have slightly clunky indexing that we may then hide behind a cleaner front end in the gradient GP class. As a for instance we might fit a model using only gradient obervations and then predict using something like

```python
gp.fit(X=None, dX=grad_inputs, y=None, dydx=grad_obs)

# return the conditional mean of the GP at x in pred_inputs
# based on the hyperpar optimisation and covar matrices
# constructed in gp.fit
gp.predict(X=pred_inputs, 'x|dx')

```

any thoughts I would be delighted to hear them via email or tweet.