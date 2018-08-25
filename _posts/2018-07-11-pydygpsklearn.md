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