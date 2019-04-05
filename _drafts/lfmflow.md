---
layout: post
title: LFM in TensorFlow
---

Slowly going to be moving most of the work I do over to TensorFlow (but maybe PyTorch...?) and so in anticipation of this switch I'm going to briefly describe an implementation of the LFM, and how it might be built using TensorFlow.

First off almost all of the models we will be interested in are going to be build on top of the <a href="https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution">`Distribution`</a> class of the <a href="">Probability</a> submodule of Tensorflow.

### Batch Shape
One of the first things to get used to is the shaping of the tensors. This is all explained in reasonable detail <a href="at https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution#broadcasting_batching_and_shapes">here</a>. As explained there a method such as `sample_n_shape` which will have

```
sample_n_shape = [n] + batch_shape + event_shape
```

### A LFM class

So what does this mean for the LFM? The LFM gives the output of `K` correlated Gaussian processes, each of which is observed at a number `N_p` of time points. So that a *single* sample from this distribution is of total size `Ntot = N_1 + ... + N_K`

```

## Defines two independent LFMs
D = [[1., 1., 1.],
     [2., 2., 2.]]

## A single instance of S is K x R
# so a batch S will be of shape
S = [ [[1, 1], [1, 1], [1, 1]],
      [[2, 2], [2, 2], [2, 2]] ]


lfm = LFM(D=D,)

```

### Multioutput GPs

We are interested in a collection of `K` scalar valued GP, each with a common dimension for their feature space, `Q` but observed at possible different points. So the input for a given shape will have the from `N_1 x F , ..., N_K x Q`, or after batching

```
x1_input = [b1, ..., bB, N1, f1, ..., fF]
x2_input = [b1, ..., bB, N2, f1, ..., fF]
...
xK_input = [b1, ..., bB, NK, f1, ..., fF]

which gets stacked together to form a single input

X_input = [b1, ..., bB, N1 +...+ NK, f1, ..., fF]
```

It seems to me that the most natural way of passing this is therefore as the stacked tensor X, with the `MultioutputGaussianProcess` class handling the shaping of these variables in a more friendly way.

Therefore

```
cov = kernel.apply(x, y)

cov.shape = broadcast([b1,...,bB], [c1,...,cC], [k1,...,kK]) + [N, M]
```