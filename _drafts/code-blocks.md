---
layout: post
title: "Code Block"
---

A code block

```
function test() {
  console.log("notice the blank line before this function?");
}
```

A Python codeblock 

```Python
import numpy as np

def f(x):
  return x**2
```

A Python codeblock with PyGments syntax highlighting
<div class="syntax">

<div class="highlight"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="kn">as</span> <span class="nn">np</span>

<span class="k">class</span> <span class="nc">MyClass</span><span class="p">:</span>
    <span class="k">def</span> <span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span><span class="n">name</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">name</span><span class="o">=</span><span class="n">name</span>
</pre></div>

</div>

And some terminal output


```
Python 3.6.0 (default, Jul 21 2017, 22:32:58) 
[GCC 4.2.1 Compatible Apple LLVM 8.1.0 (clang-802.0.42)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> from sympy import *
>>> f = Function('f')
>>> x = Symbol('x')
>>> I = integral(f(x))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'integral' is not defined
>>> I = Integral(f(x))
>>> pprint(I)
⌠        
⎮ f(x) dx
⌡        
```




