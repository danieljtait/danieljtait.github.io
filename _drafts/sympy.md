---
layout: post
title: "Symbolic Integrals in Sympy"
---

In work involving nested integrals of Gaussian processes I have often needed to solve integrals of the form

$$
\int_0^s e^{-(t-\sigma)^2} \operatorname{d}\sigma,
$$

as well as nested integrals of this form such as

$$
\int_0^{s} \int_0^{\sigma_1} e^{-(t-\sigma_1)^2} \operatorname{d}\sigma_0 \operatorname{d}\sigma_1 = \int_0^t \sigma e^{-(t-\sigma)^2} \operatorname{d}\sigma.
$$

While we could solve these integrals by hand it would be good if we could automate this procedure. With that in mind we turn to a computer algebra system (CAS) to automate this procedure and since much of the rest of my work is in Python I opted to use the open source SymPy, unfortunately I soon ran in to a problem.

```
Python 3.6.0 (default, Jul 21 2017, 22:32:58) 
[GCC 4.2.1 Compatible Apple LLVM 8.1.0 (clang-802.0.42)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import sympy
>>> sigma, s, t = sympy.symbols('sigma s t')
>>> integral = sympy.integrate(sympy.exp(-(t-sigma)**2), (sigma,0,s))
>>> sympy.pprint(integral)
√π⋅erf(t)   √π⋅erf(s - t)
───────── + ─────────────
    2             2      
```

That seems encouraging, and indeed we can differentiate it to recover the original integrand with 

```
>>> sympy.pprint(integral.diff(s).subs({s: sigma}))
         2
 -(σ - t) 
ℯ         
```

But now lets try the second equation, after a reasonably long period I get the output

```
>>> integral2 = sympy.integrate(sigma*sympy.exp(-(t-sigma)**2),(sigma,0,s))
>>> sympy.pprint(integral2)
     s                 
     ⌠                 
   2 ⎮      2          
 -t  ⎮    -σ   2⋅σ⋅t   
ℯ   ⋅⎮ σ⋅ℯ   ⋅ℯ      dσ
     ⌡                 
     0                 
```

which is SymPy's way of communicating the fact that it was unable to solve the integral. However, on closer examination of the input we of course not that what SymPy has done is extract the factor $\exp(-t^2)$ which does not dependnt on the variable of integration, it turns out this seemingly helpful step is actually where the problem lies, but first a force fix.

### Forced Integration by Parts
