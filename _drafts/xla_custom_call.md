---
layout: post
title: Custom calls in XLA
---

This is the first in a series of posts on using Jax
and the *X*ccelerated *L*inear *A*lgebra, a domain-specic
compiler for speeding up the linear algebra operations that
form the heart of modern machine learning. 

This first post demonstrates a particularly laboroius way of subtracting 
two floats... enjoy! :D 

## Registering a CustomCall

This journey began with a general interest in how I could use the
 magic of JAX on existing code written purely in C++. Vaguely aware
 that this should be possible I made my way to the 
 [documentation](https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html)
 of JAX primatives, these are the basic computational building blocks of
 JAX which works by combining these primatives into a graph of XLA operations.
 Unfortunately while the documentation suggests that
 
 > However, XLA includes a CustomCall operation that can be used to encapsulate arbitrary functionality defined using 
 > C++.

## First attempt
The JAX documents on the steps necessary to create and bind a primative and
supply information on its implementations are pretty good, so I will skip those
steps and go right to the point where we wish to register our primitive for
just-in-time compilation (JIT)
<pre class="prettyprint lang-python">
    # create a primitive for our custom function
    subtract_f32_p = core.Primitive("subtract_f32_p")  # Create the primitive
    
    def subtract_f32_prim(x, y):
        return subtract_f32_p.bind(x, y)
    
    # ... then necessary to define a concrete and abstract implementation
    # of the primitive 
    
    def subtract_float_32_xla_translation(c, xc, yc):
        return c.CustomCall(b'test_subtract_f32',
                            operands=(xc, yc),
                            shape_with_layout=xla_client.Shape.array_shape(jnp.dtype(jnp.float32), (), ()),
                            operand_shapes_with_layout=(
                                xla_client.Shape.array_shape(jnp.dtype(jnp.float32), (), ()),
                                xla_client.Shape.array_shape(jnp.dtype(jnp.float32), (), ())
                            ))
    
    # Now we register the XLA compilation rule with JAX...
    xla.backend_specific_translations['cpu'][subtract_f32_p] = subtract_float_32_xla_translation
    
    # and run it!
    jax.api.jit(subtract_f32_prim)(1.25, 0.5)
 </pre>

Ok so lets run the script and see what happens...

<pre class="prettyprint lang-shell">
    Unable to resolve runtime symbol: `_test_subtract_f32'.  Hint: if the symbol a custom call target, make sure you've registered it with the JIT using XLA_CPU_REGISTER_CUSTOM_CALL_TARGET.
    Bus error: 10
</pre>

Oh no! Of course this isn't surprising, we haven't done anything to 
tell this program anything about the C++ function `test_subtract_f32` or
where to find it -- we haven't even imported into Python at this point!
So how do we do this? Unfortunately the details of this are currently 
*very* sparse, and heading over to the [XLA documentation](https://www.tensorflow.org/xla/custom_call) 
was not  patricularly enlightening.

As is so often the case the answers are not to be
found in the documents, but in the source code if you know
where to look. In this case the right place to look is 
`tensorflow/compiler/xla/python/xla_client_test.py`
and so this first post essentially lifts this basic test function 
out of the larger test suite, and in future posts we will show how 
to register this more fully with JAX for taking gradients, etc.

## Defining our C++ function 
Our first goal is to write our function in C++ that
we would like to use in our larger python project while
still allowing for JIT using the XLA compiler. The initial
process of writing the function will be done with [Cython](https://cython.org/).
The scipt for defining the function is contained in the Tensorflow
xla compiler test suite and in full is

#### [`tensorflow/compiler/xla/python/test_custom_call.pyx`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/python/custom_call_for_test.pyx)

<pre class="prettyprint lang-python">
    # cython: language_level=2
    # distutils: language = c++
    
    # Test case for defining a XLA custom call target in Cython, and registering
    # it via the xla_client SWIG API.
    from cpython.pycapsule cimport PyCapsule_New
    
    cdef void test_subtract_f32(void* out_ptr, void** data_ptr) nogil:
        cdef float a = (<float*>(data_ptr[0]))[0]
        cdef float b = (<float*>(data_ptr[1]))[0]
        cdef float* out = <float*>(out_ptr)
        out[0] = a - b
    
    
    cpu_custom_call_targets = {}
    
    cdef register_custom_call_target(fn_name, void* fn):
        cdef const char* name = "xla._CUSTOM_CALL_TARGET"
        cpu_custom_call_targets[fn_name] = PyCapsule_New(fn, name, NULL)
    
    register_custom_call_target(b"test_subtract_f32", <void*>(test_subtract_f32))
</pre>

Most of this is pretty standard Cython, but it is the final few lines that
provide the `xla` specific details. If we write a short setup script

#### [`/setup.py`](https://github.com/danieljtait/flax_xla_adventures/blob/master/initial_custom_call/setup.py)
<pre class="prettyprint lang-python">
    
    from distutils.core import setup
    from Cython.Build import cythonize
    
    setup(
        ext_modules=cythonize("custom_call_for_test.pyx")
    )
</pre>

and then build the module

<pre class="prettyprint lang-shell">
    $ python setup.py build_ext --inplace
</pre>

we can import this function and check out the `cpu_custom_call_targets` 
dictionary that has been created in the `custom_call_for_test` module

<pre class="prettyprint lang-python">
    >>> import custom_call_for_test
    >>> custom_call_for_test.cpu_custom_call_targets
    {b'test_subtract_f32': <capsule object "xla._CUSTOM_CALL_TARGET" at 0x10dd75f60>}
</pre>

Which provides a name and a [`PyCapsule`](https://docs.python.org/2/c-api/capsule.html)
encapsulating the pointer to our low level function. Ok now we have a C++ function 
inside a capsule and have imported it into python, how do we tell XLA about it?

### Registering the CustomCall
Turns out that the actual registration process is very simple now:

<pre class="prettyprint lang-python">
    import custom_call_for_test
    from jaxlib import xla_client
        
    # register the function
    for name, fn in custom_call_for_test.cpu_custom_call_targets.items():
        xla_client.register_cpu_custom_call_target(name, fn)
</pre>

That's it! The xla_client now knows about the function named `test_subtract_f32`.
We can run the complete test code now as

#### [`/test.py`](https://github.com/danieljtait/flax_xla_adventures/blob/master/initial_custom_call/test.py)
<pre class="prettyprint lang-python">
    import jax.numpy as jnp
    from jaxlib import xla_client
    import custom_call_for_test
    
    # register the function
    for name, fn in custom_call_for_test.cpu_custom_call_targets.items():
        xla_client.register_cpu_custom_call_target(name, fn)
    
    c = xla_client.ComputationBuilder('comp_builder')
    
    c.CustomCall(b'test_subtract_f32',
                 operands=(c.ConstantF32Scalar(1.25), c.ConstantF32Scalar(0.5)),
                 shape_with_layout=xla_client.Shape.array_shape(jnp.dtype(jnp.float32), (), ()),
                 operand_shapes_with_layout=(
                     xla_client.Shape.array_shape(jnp.dtype(jnp.float32), (), ()),
                     xla_client.Shape.array_shape(jnp.dtype(jnp.float32), (), ())
                 ))
    
    compiled_c = c.Build().Compile()
    result = xla_client.execute_with_python_values(compiled_c, ())
    print("Result: {} Expected: {}".format(result, 0.75))
</pre>

Or add the registration step to our more complete module to demonstrate that
we can now JIT our function.

Full codes to run this simple example are available on [my github](). Future 
posts will discuss how to also perform forward differentiation of our
newly defined function.