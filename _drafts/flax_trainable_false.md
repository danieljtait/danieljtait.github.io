---
layout: post
title: <code>trainable == False</code> in Flax (part i)
---

## Introduction 

One of the nicest features of the GPFlow software
<a href="https://github.com/GPflow/GPflow">GPFlow</a> 
is the ability to create an
instance of your model, and then set certain parameters
to be fixed during all or part of training. Typically
this can be done in a line of code with something like

<pre class="prettyprint lang-python">
gpflow.utilities.set_trainable(model.param, False)
</pre>

In contrast the more functional nature of models in Jax/Flax
mean that we shift away from considering a single instantiated
instance of our model object, and instead most of the work
is done by composable functional layers in the `nn.Module` class,
and these are lightly wrapped inside of a `nn.Model` class. 
The model class behaves like frozen `dataclass`, so that any
time we watn to modify a parameter we actually have to create a 
new instance of the model with the unmodified parameters copied over.

In the following we briefly sketch a functional approach to 
recreating the general idea of fixing some parameters and not 
modifying them during training using models constructed in Flax. 

## Fixing the parameters of a `flax.nn.Module`

First lets create a simple
multiplication layer with parameter `a`:

<pre class="prettyprint lang-python">
class Layer(flax.nn.Module):
    def apply(self, x, a_fixed=False, a_init=jax.nn.initializers.ones):
        a = self.param('a', (1, ), a_init)
        if a_fixed:
            _default_key = random.PRNGKey(0)
            a = a_init(_default_key, (1, ))
        return a * x
</pre>
Note that this code *always* initalizes the parameter `a`, but if we
also pass the argument `a_fixed` then the parameter never reaches the
return. This has the advantage of allowing for `a` to still be in our 
`model.params`, and therefore we can for example modify the fixed
parameters using the `replace` method, but it means that in any gradient
based training methods the parameter will never reach the tape. There
is a cost of this disconnect which we will return to later.


Now we will imagine this is called inside of some larger model
<pre class="prettyprint lang-python">
class MyModel(flax.nn.Module):
    def apply(self, x, **kwargs):
        x = Layer(x, **kwargs.get('layer_kwargs', {}), name='layer')
        return x
</pre>
Of course keeping track of all of these `kwargs` every single time we
make use of a `model.apply(...)` is going to be a real headache! 
One method of handling this is to use the `.partial` method of the
`nn.flax.Module`, as demonstrated over at the 
[flax docs](https://flax.readthedocs.io/en/latest/notebooks/flax_intro.html#Model).

Using this we now just have a one time burden of specifying a larger set of 
`kwargs`, then using the partial method to create a new definition of our model, 
and then optionally writing a model creation method that is aware of these 
different definitions. The result is something like

<pre class="prettyprint lang-python">
free_kwargs = {'layer_kwargs': {'a_fixed': False}}
fixed_kwargs = {'layer_kwargs': {'a_fixed': True}}

# use partial to fix the initial functions
free_model_def = MyModel.partial(**free_kwargs)
fixed_model_def = MyModel.partial(**fixed_kwargs)

def create_model(model_def, key, input_specs):
    x, init_params = model_def.init_by_shape(key, input_specs)
    return flax.nn.Model(model_def, init_params)
</pre>

As we can see the fixed model still has `a` as a parameter
<pre class="prettyprint lang-shell">
>>> fixed_model.params
{'layer': {'a': DeviceArray([1.], dtype=float32}}
</pre>
Now if we define a test function and take gradients we find
<pre class="prettyprint lang-python">
def loss_fn(model):
    x = jnp.ones(*input_shape_and_dtype[0])
    y = model(x)
    return jnp.mean(y ** 2)

free_model_grad = jax.grad(loss_fn)(free_model)
fixed_model_grad = jax.grad(loss_fn)(fixed_model)

assert(loss_fn(free_model) == loss_fn(fixed_model))
assert(free_model_grad.params['layer']['a'] == 2)
assert(fixed_model_grad.params['layer']['a'] == 0)
</pre>

## Big caveat
Some of you may have noticed the issue which will occur
in the following use case

<pre class="prettyprint lang-python">
new_fixed_model = fixed_model.replace(
    params={'layer': {'a': 3.14*jnp.ones([1])}})

assert(new_fixed_model(x) != fixed_model)
</pre>
will raise an `AssertionError`! Because of our earlier comment
about the disconnect between the parameter and the layer output.
Instead what we want is actually a new model definition

<pre class="prettyprint lang-python">
new_fixed_kwargs = {
    'layer_kwargs':
        {'a_fixed': True, 'a_init': lambda key, shape: 3.14*jnp.ones([1])}}
</pre>

we will return to this point in part ii of this series. 

## Further comments and ToDos
In contrast to gpflow we are not able to take an existing model instance, and change a previously
free parameter to a fixed one -- ultimately it doesn't seem intended behaviour in `flax`
for an instance to persist over long periods of the implementation. Instead we can quote the docs

 > A model instance is callable and functional (e.g. changing
 > parameters requires a new model instance.)

This takes some getting used to at first if you are coming from a more object orientated perspective,
and it does make the process of model creation feel quite different. While it would also be possible to
wrap this inside of more syntactic sugar, in the way that the current GPFlow parameters are themselves
wrapping the base `Module` class with its variable tracking in `Tensorflow`, it is probably best not to
do this. Instead it seems like it would probably be preferable to have the user need to write potentially
quite a large number of lines to make their model behave the way they want to, so long as each of these
lines was simple and their composition intuitive and easy to follow.  

In the next post we will show how to better adapt this method to handle changing the fixed parameter, 
and then in the next few weeks we will demonstrate some examples of various models coded in Flax compared 
to their GPFlow equivalents to give some sense of how these different approaches manifest for the 
user/model builder.

