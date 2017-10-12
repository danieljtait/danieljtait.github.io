---
layout: default
title: Bayesian Inference in ODEs
---

Parameter inference in models described by a set of differential equations is particularly challenging because the absence of closed form expressions for the posterior distributions of the model parameters - but the usual means of overcoming this issue by means of Monte Carlo methods is made challenging because in a naive approach each proposed new parameter value would require an associated solution of the differential equation which can become prohibitvely expensive, in this post I will give an introduction of the method proposed in Calderhead et al. (2008) introducing the model description at the same time indicating how to fit the model using the code contained in <a href="somepage">[link to github]</a>

## Model specification
The data consists of a set of points $\mathbf{Y} = \left( \mathbf{y}_1, \ldots, \mathbf{y}_n \right)$ observed at times $\mathbf{t} = (t_1,\ldots,t_n)$ with basic observation model

$$
\mathbf{y}(t) = \mathbf{x}(t) + \epsilon(t), \qquad \epsilon(t) \sim \mathcal{N}\left(\epsilon(t)|\mathbf{0}, \sigma^2 \mathbf{I} \right).
$$

The dynamics of the $K$-dimensional state variable $\mathbf{x}(t)$ are described by some ordinary differential equation

$$
\dot{\mathbf{x}}(t) = f(\mathbf{x}(t), \mathbf{\theta})
$$

### Lotka-Volterra Example

This model has a very simple generative structure, and the following code will simulate the model for the case where

$$
\begin{align*}
\dot{x}_1 &= x_1(\theta_1 - \theta_2 x_2) \\
\dot{x}_2 &= -x_2(\theta_3 - \theta_4 x_1).
\end{align*}
$$

<script src="https://gist.github.com/danieljtait/9b7c38844855d74053e385b0040c497f.js"></script>

### Decoupled state variable
As mentioned above the state variable is drawn from the distribution

$$
\mathbf{X} \sim p(\mathbf{X}\,|\,\mathbf{Y}, \sigma, \phi)
$$

which has no dependence on the model parameters, but also no dependence on the structural properties of the ordinary differential equation at all. It is simply a Gaussian process regression with observations $\mathbf{X}$.

An extension in ... introduces a mixture of experts approximation to the density

$$
p(\dot{\mathbf{x}}_k \, | \, \mathbf{X}, \theta, \phi, \gamma_k ) \propto 
$$

after marginalising the resulting joint density

$$
p() \propto \exp\left\{-\frac{1}{2} \mathbf{x}_k^T\mathbf{C}^{-1}_{\phi_k}\mathbf{x}_k + \left( \right)\left( \right)^{-1}\left(\mathbf{f}_k-\mathbf{m}_k\right)\right\}
$$

but even in the case where $\mathbf{f}_k$ is a linear function of $\mathbf{X}$ we have a Gaussian distribution and as mentioned above it is possible even in the linear case to have non Gaussian densities for the resulting state variables.

<pre>
  <code class="python">
    def func():
        pass
  </code>
</pre>

```python
def func():
    pass
```

<script src="https://gist.github.com/danieljtait/4c1cf747dd078442d38f362a7f2cdb79.js"></script>

### Kubo Oscillator

Of course our above discussion makes clear that this is not an appropriate approach for this example, on the otherhand it is suggestive of the underlying point that the structure above may not capture the underlying structure of these dynamic models even in the simple linear case and should serve as a cautionary tale with to prevent automatic application of these methods, and as a motivation for construction of methods which better preserve this structure without requiring the complete solution of the model for each proposed parameter value. To demonstrate the discussion above we will carry out the method described in this post to the following simple ODE model

$$
\begin{bmatrix}
\dot{x}_1 \\ \dot{x}_2 
\end{bmatrix} = \begin{bmatrix} 0 & -\theta \\ \theta & 0 \end{bmatrix}\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
$$

now clearly conditional on the state variable $\mathbf{X}$, the posterior of $\theta$ has a discrete support. In fact we can solve explicitly to give

$$
X(t) = \cos(\theta t)x_0 - \sin(\theta t)y_0, \qquad Y(t) = \sin(\theta t)x_0 - \cos(\theta t)y_0
$$

now if put a Gaussian prior on the initial condition then this conditional is a Gaussian process, this is a general feature of these models, the structural parameters of the ODE enter as kernel hyper parameters