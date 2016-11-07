

"""
This package includes modules to create the medium and sensor configuration required for the simulations.
The recommended way to build these objects::

    from smrt import make_snowpack, sensor_list

    sp = make_snowpack([1000], density=[300], microstructure_model='sticky_hard_spheres', radius=[0.3e-3], stickiness=0.2)

    radiometer = sensor_list.amsre()

Note that the function :py:func:`~smrt.inputs.make_medium.make_snowpack` and the module :py:mod:`~smrt.inputs.sensor_list` is directly imported from smrt, which is convenient but they effectively lie 
in the package :py:mod:`smrt.inputs`. They could be imported using the full path as follows::

    from smrt.inputs.make_medium import make_snowpack
    from smrt.inputs import sensor_list

    sp = make_snowpack([1000], density=[300], microstructure_model='sticky_hard_spheres', radius=[0.3e-3], stickiness=0.2)

    radiometer = sensor_list.amsre()

Extension of the modules in the `inputs` package is welcome. This is as simple as adding new functions in the modules (e.g. in :py:mod:`~smrt.inputs.sensor_list`) or
adding a new modules (e.g. `my_make_medium.py`) in this package and use the full path import.

"""
