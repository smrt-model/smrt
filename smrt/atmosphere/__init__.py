""" This directory contains different options to represent the atmosphere, that is the upper boundary conditions 
of the radiation transfer equation.

This part is currently not fully developed but should work for an isotropic atmosphere.

Example::

    from smrt.atmosphere.basic import ConstantAtmosphere

    atmosphere = ConstantAtmosphere(tbdown=2.7, tbup=2.7, trans=0.998)

The API is subject to change.


 """  # html documentation