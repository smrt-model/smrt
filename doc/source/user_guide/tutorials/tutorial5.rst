##################################
Extending SMRT
##################################

There are different ways to extend SMRT, here we address the case of ice
permittivity.

Open the smrt/permittivity/ice.py file in an editor to see how it looks
like: permittivity functions are defined as normal python functions with
several arguments. There is some rules or some tricks: - ``frequency``
is the first one and MUST be there for any permittivity function. - the
second one is often ``temperature``, this is recommended. - optionaly
other arguments depending on the formulation.

How SMRT know what to do with this variable number of arguments ?

We heavily use dynamical nature of python because we really want users
to define new arguments at will, without changing the core of the model
and keeping the compatibility. Here for the permittivity, the trick is
in the declaration ``@layer_properties("temperature", "salinity")`` put
just before the function declaration. This tells SMRT that this function
needs to temperature and salinity arguments that are taken from the
layer for which we want to compute the permittivity. The important point
is that **any new arguments can be defined without changing anything in
SMRT core**.

Example:

.. code:: ipython3

    from smrt import make_model, make_snowpack, sensor_list
    
    from smrt.core.layer import layer_properties

.. code:: ipython3

    # let's defined a new function
    
    @layer_properties("temperature", "potassium_concentration")
    def new_ice_permittivity(frequency, temperature, potassium_concentration):
        return 3.1884 + 1j * (0.1 + potassium_concentration * 0.001)  # this is purely imaginative!!!!!!!!

.. code:: ipython3

    # let's defined the snowpack
    
    thickness = [10]
    density = 350
    temperature = 270
    radius = 100e-6
    
    sp = make_snowpack(thickness, 'sticky_hard_spheres',
                       density=density, radius=radius, temperature=temperature,
                       potassium_concentration=0.1,
                       ice_permittivity_model=new_ice_permittivity) # here we declare we want the new permittivity


.. code:: ipython3

    sp.layers[0].potassium_concentration

.. code:: ipython3

    sensor = sensor_list.amsre()
    m = make_model("iba", "dort")
    result = m.run(sensor, sp)
    
    # execute this code and see the last line of the error message below
    # does it make sense ? The call to the new_ice_permittivity function needs
    # potassium_concentration to be provided.
    # to fix the problem, just add potassium_concentration=5.0e-3 to the make_snowpack call and reexcute.
    # the cells.
    # Remember: "potassium_concentration" never appears in SMRT code, it is purely user-defined.
    # Any other variables (as long as it does not colleige with internal SMRT naming) is valid.
    # "K_conc", "myarg1" are valid though we strongly recommend explicit naming such as potassium_concentration 

