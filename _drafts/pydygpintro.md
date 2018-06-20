---
layout: post
title: "Introducing PydyGP"
---

Generalisation remains a fundamental problem in the implementation of modern machine learning problems and in particular the problem of embedding domain dependent information in a principled fashion. This problem has inspired by interest in *hybrid models* which aim to combine the flexibility of modern machine learning with a certain amount of mechanistic structure. This then is a quick announcement on the release of my package

`pydygp`
> a **Py**thon package for **Dy**namic systems with
> latent **G**aussian **P**rocesses

which aims to provide usable and robust implementation of a class of hybrid machine-learning algorithms which aims to achieve this goal by combining simple dynamic equations with flexible Gaussian processes to achieve the best of both worlds.

While I am still in the process of releasing the full code from my hard drive in to the light over on [github](https://github.com/danieljtait/pydygp) it has got to the point where there is now enough code moved across to now be of interest to a larger audience.

The package is hosted over on [PyPi](https://pypi.org/project/pydygp/) and so should be installable with

```bash
pip install pydygp
```

However in the short term cloning the the repository on github is probably advisable.


## Features
As a near term goal I will be primarily focused on

* (Additive) Linear latent force models
* Multiplicative latent force models

Which will more or less takes this class of models right up to the current state of the art, but it is to be hoped this list continues to expand as the art progresses.

## Further Info
I will be continuously updating the [documentation](https://pydygp.readthedocs.io/en/latest/) and in particular would point interested parties in the direction of the [user guide](https://pydygp.readthedocs.io/en/latest/user/index.html) and examples contained within.

## Contribution
The project is very much in its infancy but if you are interested in contributing anything from documentation, tutorials and examples or any relevant modules do get in touch at <`package name`>`@gmail.com`.
