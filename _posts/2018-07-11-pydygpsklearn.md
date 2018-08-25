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