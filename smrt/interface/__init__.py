""" This module contains different type of boundary conditions between the layers.
Currently only flat interfaces are implemented. 

.. admonition::  **For developers**

    All the different type of interface must defined the methods: `specular_reflection_matrix` and `coherent_transmission_matrix`.

    It is currently not possible to implement rough interface, a (small) change is needed in DORT. Please contact the authors.

"""
