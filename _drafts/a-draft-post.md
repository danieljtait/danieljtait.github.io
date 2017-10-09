---
layout: post
title: "Kubo Oscillator"
---

In this post we introduce the *Kubo oscillator* as a representative example of what we shall term *multiplicative* latent Gaussian Process force models. The evenolution of the Kubo Oscillator is decribed by the differential equation

$$
\begin{bmatrix} \dot{x}_1 \\ \dot{x_2} \end{bmatrix}
= \left( \mathbf{B} + \begin{bmatrix} 0 & -\epsilon(t) \\ \epsilon(t) & 0 \end{bmatrix} \right)\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}, \tag{1}
$$
where \[ \epsilon(t) \] is a *smooth* Gaussian process, and therefore we may intepret