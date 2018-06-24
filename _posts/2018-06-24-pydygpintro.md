---
layout: post
title: "Introducing PydyGP"
---

Generalisation remains a fundamental problem in the implementation of modern machine learning problems and in particular the problem of embedding domain dependent information in a principled fashion. This problem has inspired my interest in *hybrid models* which aim to combine the flexibility of modern machine learning with a certain amount of mechanistic structure.

<img src="/assets/hybridmodelling/hybridmodelling.png" class="center" width=400px />

This then is a quick announcement of the release of my package

> `PydyGP`: a **Py**thon package for **Dy**namic systems with
> latent **G**aussian **P**rocesses

The aim is to provide a usable and robust implementation of a class of hybrid machine-learning methods for systems with time dependent features by combining simple dynamic equations with flexible Gaussian processes in an attempt to achieve the best of both worlds. While I am still in the process of releasing the full code from my hard drive in to the light over on [github](https://github.com/danieljtait/pydygp) it has got to the point where there is now enough code moved across to be of interest to a larger audience.

## Where can I find `pydygp`?

The package is hosted over on [PyPi](https://pypi.org/project/pydygp/) and so should be installable with

```bash
pip install pydygp
```

However in the short term the package is likely to grow quite rapidly and so cloning the the repository on github is probably advisable.


## Features
As a near term goal I will be primarily focused on:

* **(Additive) Linear latent force models**: These models are described in <a href="#ref1">[1]</a>, they combine simple linear ODEs with a flexible Gaussian process term

$$
\begin{align}
  \dot{\mathbf{x}}(t) = \mathbf{A}\mathbf{x}(t) + \mathbf{S}\mathbf{g}(t),
\end{align}	
$$

* **Multiplicative latent force models**: This is an extension of the latent force model to allow interactions between the state variables and the latent forces

$$
\begin{align}
  \dot{\mathbf{x}}(t) = \left(\mathbf{A}_0 + \mathbf{A}_r g_r(t) \right)\mathbf{x}(t)
\end{align}	
$$

Where in both cases $\mathbf{g}$ and $\{ g_r \}$ represent latent Gaussian processes.

One approach to fitting the second class of models is the adaptive gradient matching methods introduced in <a href="#ref2">[1]</a> and so at a future date I may adapt the existing code to handle the Metropolis-Hastings fitting of more general nonlinear ODE models using the adaptive gradient approach. However is worth noting that these models are philisophically quite different, often being very well specified mechanistic models with a small set of random parameters, and unlike the two evolution equations introduced above where the GP forces *drive* the equation, the use of the GP terms in the adaptive gradient matching approach is as an interpolater of the latent states used to avoid the explicit solution of the ODE.

## Further Info
I will be continuously updating the [documentation](https://pydygp.readthedocs.io/en/latest/) and in particular would point interested parties in the direction of the [user guide](https://pydygp.readthedocs.io/en/latest/user/index.html) and examples contained within.

## Contribution
The project is very much in its infancy but if you are interested in contributing anything from documentation, tutorials and examples or any relevant modules do get in touch at <`package name`>`@gmail.com`.

## References

1. <a name="ref1"></a>Dondelinger F, Filippone M, Rogers S, Husmeier D *ODE parameter inference using adaptive gradient matching with Gaussian processes*
2. <a name="ref2"></a>Alvarez M, Luengo D, Lawrence N *Latent Force Models*, PMLR 5:9--16, 2009