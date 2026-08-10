##############
Extending SMRT
##############

**Goal**: understand how to create new functions to extend SMRT locally.

For information on how to contribute to SMRT, please refer to `Developer Guidelines <../../developer/index.html>`_.

This guide takes as example the definition of a new ice
permittivity function.

Open the ``smrt/permittivity/ice.py`` file in an editor to see how it looks
like: permittivity functions are defined as normal python functions with
several arguments but there are some specific rules:

- ``frequency`` is the first argument and MUST be there for any permittivity function.
- the second argument is often ``temperature``, this is recommended.
- there may be other optional arguments depending on the formulation.

We heavily use dynamical nature of Python because we really want users
to be able to define new arguments at will, without changing the core of the model. For the permittivity, the trick is
in the declaration ``@layer_properties("temperature", "salinity")`` put
just before the function declaration. This tells SMRT that this function
needs two  arguments (``temperature`` and ``salinity``, in addition to ``frequency``) that are automatically taken from
the layer for which we want to compute the permittivity. The important point is that **any new arguments can be defined
without changing anything in SMRT core**.

Let's define a new arbitrary permittivity function.

.. code:: ipython3

    from smrt.core.layer import layer_properties

    @layer_properties("temperature", "potassium_concentration")
    def new_ice_permittivity(frequency, temperature, potassium_concentration):
        return 3.1884 + 1j * (0.1 + potassium_concentration * 0.001)

Create a snowpack to test it:

.. code:: ipython3

    from smrt import make_model, make_snowpack, sensor_list

    thickness = [10]
    density = 350
    temperature = 270
    radius = 100e-6

    sp = make_snowpack(thickness, 'sticky_hard_spheres',
                       density=density, radius=radius, temperature=temperature,
                       potassium_concentration=0.1,
                       ice_permittivity_model=new_ice_permittivity) # here we declare we want the new permittivity


Make sure the snowpack layers have a ``potassium_concentration``:

.. code:: ipython3

    sp.layers[0].potassium_concentration

You can now use the new function for simulations:

.. code:: ipython3

    sensor = sensor_list.amsre()
    m = make_model("iba", "dort")
    result = m.run(sensor, sp)

``potassium_concentration`` never appears in SMRT code, it is purely user-defined.
Any other variables (as long as it does not collide with internal SMRT variable names) is valid.

**Note:** A more complex definition of an ice permittivity model for wet snow is described in
`this Github issue <https://github.com/smrt-model/smrt/issues/17>`_.

Recap:
======

SMRT is build in a way that users can create their own functions for most building blocks of the simulator without
having to modify the core of the library. If you need to implement a new function:

- have a look at the `API Reference <../../api/index.html>`_ to understand where to find similar functions;
- check the code to mimic the structure of similar functions;
- use e.g. ``layer_properties`` to pass your own arguments;
- if you believe your code can be useful to others, refer to `Developer Guidelines <../../developer/index.html>`_ to
see how to include it to SMRT following standard practices.
