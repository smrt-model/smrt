# coding: utf-8

"""
Wraps the original HUT matlab using SMRT framework.

To use this module, extra installations are needed:

    * Gets HUT. Decompresses the archive somewhere on your disk.
    * In the file snowemis_nlayers, changes the 6 occurrences of the "do" variable into "dos" because it causes a syntax error in Octave.
    * Installs the oct2py module using :code:`pip install oct2py` or :code:`easy_install install oct2py`.
    * Installs Octave version 3.6 or above.
    * For convenience, sets the HUT_DIR environment variable to point to HUT path. This path can also be programmatically set with :py:func:`set_hut_path`.

In case of problem, checks the instructions given in http://blink1073.github.io/oct2py/source/installation.html.

"""

import os
from collections.abc import Sequence

import numpy as np
from oct2py import octave

from smrt.core.globalconstants import DENSITY_OF_ICE, FREEZING_POINT
from smrt.core.result import Result, concat_results

# python-space path to memls
_hut_path = None


def set_hut_path(path):
    """
    Sets the path where HUT archive has been uncompressed, i.e. where the file `memlsmain.m` is located.

    Args:
        path: Path to the HUT directory.
    """
    global _hut_path

    if path != _hut_path:
        # octave.restoredefaultpath() # risk of bad interference with DMRT_QMS and MEMLS
        octave.addpath(path)
        octave.addpath(os.path.dirname(__file__))
        _hut_path = path


try:
    # set
    set_hut_path(os.environ["HUT_DIR"])
except KeyError:
    pass


def run(sensor, snowpack, ke_option=0, grainsize_option=1, hut_path=None):
    """
    Calls HUT for the snowpack and sensor configuration given as argument. Any microstructure model that defines the "radius" parameter is valid.

    Args:
        sensor: Sensor configuration.
        snowpack: Snowpack description.
        ke_option: Option for HUT snowemis_nlayers.m code.
        grainsize_option: Option for HUT snowemis_nlayers.m code.
        hut_path: Optional path to HUT.

    Returns:
        Result object.
    """

    if hut_path is not None:
        set_hut_path(hut_path)

    if isinstance(snowpack, Sequence):
        result_list = [run(sensor, sp, ke_option=ke_option, grainsize_option=grainsize_option) for sp in snowpack]
        return concat_results(result_list, ("snowpack", range(len(snowpack))))

    if snowpack.substrate is not None:
        Tg = snowpack.substrate.temperature
        roughness_rms = getattr(snowpack.substrate, "roughness_rms", 0)
        soil_eps = snowpack.substrate.permittivity(sensor.frequency, Tg)
    else:
        Tg = FREEZING_POINT
        roughness_rms = 0
        soil_eps = 1

    snow = []  # snow is a N Layer (snowpack+soil) row and 8 columns. Each colum has a data (see snowemis_nlayer)
    enough_warning = False

    for lay in snowpack.layers:
        density = lay.frac_volume * DENSITY_OF_ICE
        snow.append(
            (
                lay.temperature - FREEZING_POINT,
                lay.thickness * density,
                2000 * lay.microstructure.radius,
                density / 1000,
                lay.liquid_water,
                lay.salinity,
                0,
                0,
            )
        )
        if lay.salinity and enough_warning:
            print("Warning: salinity in HUT is ppm")
            enough_warning = True
    # ground
    snow.append((Tg - FREEZING_POINT, 0, 0, 0, 0, 0, roughness_rms, soil_eps))

    thetad = np.degrees(sensor.theta)
    TbV = [
        octave.snowemis_nlayer(otulo, snow, sensor.frequency / 1e9, 0, ke_option, grainsize_option) for otulo in thetad
    ]
    TbH = [
        octave.snowemis_nlayer(otulo, snow, sensor.frequency / 1e9, 1, ke_option, grainsize_option) for otulo in thetad
    ]

    coords = [("theta", sensor.theta), ("polarization", ["V", "H"])]

    return Result(np.vstack((TbV, TbH)).T, coords)
