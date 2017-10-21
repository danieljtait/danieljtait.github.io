---
layout: default
title: Bayesian Inference in ODEs
---


Bayesian inference in dynamic models described by a set of parameterised ordinary differential equations (ODEs) is complicated by the requirement of recalculating the trajectory for different values of the model parameters. In general this will require a large number of numerical implement ... which becomes increasingly expensive, and furthermore for most applications and accurate computation of the trajectory is redundant. The goal is then to derive a faster method of producing a trajectory that

> * *provides a quick method of computing an (approximate) trajectory of the driving ODE*
> * *sufficiently captures the dependence of the trajectory on the model parameters*

In this post I will give an introduction of the method proposed in Calderhead et al. (2008) for addressing this particular problem and how it deals with the two key points described above. A long the way I will illustrate the method with a fully working example allowing the reader to implement a version of the model.

introducing the model description at the same time indicating how to fit the model using the code contained in <a href="somepage">[link to github]</a>

particularly challenging because 

of the absence of closed


form expressions for the posterior distributions of the model parameters - but the usual means of overcoming this issue by means of Monte Carlo methods is made challenging because in a naive approach each proposed new parameter value would require an associated solution of the differential equation which can become prohibitvely expensive,


### Basic Model specification
The modelling scenario we consider consists of a sequence of observed variables $\mathbf{Y} = \left( \mathbf{y}_1, \ldots, \mathbf{y}_n \right)$ representing a set of noisy observations from the continuous trajectory $\mathbf{X}(t)$ observed at the discrete time points $\mathbf{t} = (t_1,\ldots,t_n)$, that is we specify a basic observation model of the form

$$
\mathbf{y}(t) = \mathbf{x}(t) + \epsilon(t), \qquad \epsilon(t) \sim \mathcal{N}\left(\epsilon(t)|\mathbf{0}, \sigma^2 \mathbf{I} \right).
$$

the realisations of the process are assumed to be $D$-dimensional and we assign an indepedent noise prior for each dimension and so we write

$$
\mathbf{y}_k = \mathbf{x}_k + \epsilon(t), \qquad \epsilon_k(t) \sim \mathcal{N}\left( \epsilon(t) | \mathbf{0}, \sigma_k \right).
$$

In this formulation the true trajectory is assumed to be latent and described by a parameterised ODE

$$
\dot{\mathbf{X}}(t) = f(\mathbf{X}(t), \mathbf{\theta}).
$$

### Lotka-Volterra Example

To make the specification described above explict I will detour briefly to describe an explicit example of such a model. The dynamics of the latent trajectory will be given by an example of a [Lotka-Volterra](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations)  system of differential equations

$$
\begin{align*}
\dot{x}_1 &= x_1(\theta_1 - \theta_2 x_2) \\
\dot{x}_2 &= -x_2(\theta_3 - \theta_4 x_1).
\end{align*}
$$

<script src="https://gist.github.com/danieljtait/9b7c38844855d74053e385b0040c497f.js"></script>

At this point our model would be completely specified after we have defined a prior distribution over each of the model parameters. 

![picture]( {{ "/assets/Figure_1.png" | absolute_url }} )


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