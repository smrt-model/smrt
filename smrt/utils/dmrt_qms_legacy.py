# coding: utf-8

""" Wrapper to the original DMRT_QMS matlab code using the SMRT framework. To use this module, extra installation are needed:

 * get DMRT_QMS from http://web.eecs.umich.edu/~leutsang/Available%20Resources.html and extract the model somewhere

 * install the oct2py module using :code:`pip install oct2py` or :code:`easy_install install oct2py`

 * install Octave version 3.6 or above.

 * for convenience you can set the DMRT_QMS_DIR environment variable to point to DMRT-QMS path. This path can also be programmatically set with and use :py:func:`set_dmrt_qms_path` function.

In case of problem check the instructions given in http://blink1073.github.io/oct2py/source/installation.html

You may also want to increase the number of streams in passive/DMRT_QMS_passive.m

"""

import os
from collections.abc import Sequence
from collections import namedtuple

import numpy as np

from oct2py import octave, Struct

from smrt.core.snowpack import Snowpack
from smrt.core.result import PassiveResult, concat_results
from smrt.core.sensitivity_study import SensitivityStudy
from smrt.core.globalconstants import DENSITY_OF_ICE, GHz


# python-space path to dmrt_qms.
_dmrt_qms_path = None


def set_dmrt_qms_path(path):
    """set the path where dmrt_qms archive has been uncompressed, i.e. where the file `dmrt_qmsmain.m` is located."""
    global _dmrt_qms_path

    if path != _dmrt_qms_path:
        # octave.restoredefaultpath() # risk of bad interference with MEMLS
        octave.addpath(os.path.join(path, 'passive'))
        octave.addpath(os.path.join(path, 'active'))
        octave.addpath(os.path.join(path, 'common'))
        octave.addpath(os.path.dirname(__file__))
        _dmrt_qms_path = path


try:
    # set
    set_dmrt_qms_path(os.environ['DMRT_QMS_DIR'])
except KeyError:
    pass


def run(sensor, snowpack, dmrt_qms_path=None, snowpack_dimension=None, full_output=False):
    """call DMRT-QMS for the snowpack and sensor configuration given as argument. The :py:mod:`~smrt.microstructure_model.sticky_hard_spheres` microstructure model 
    must be used.

    :param snowpack: describe the snowpack.
    :param sensor: describe the sensor configuration.
    :param full_output: determine if ks, ka and effective permittivity are return in addition to the result object
"""

    if dmrt_qms_path is not None:
        set_dmrt_qms_path(dmrt_qms_path)

    if isinstance(snowpack, SensitivityStudy):
        snowpack_dimension = (snowpack.variable, snowpack.values)
        snowpack = snowpack.snowpacks.tolist()

    if isinstance(snowpack, Sequence):
        result_list = [run(sensor, sp) for sp in snowpack]
        if snowpack_dimension is None:
            snowpack_dimension = 'snowpack', range(len(snowpack))
        return concat_results(result_list, snowpack_dimension)

    Tg = snowpack.substrate.temperature if snowpack.substrate is not None else 273.0

    rough = Struct()

    if snowpack.substrate is None:
        rough.model = 'QH'
        epsr_ground = complex(1.0, 0.0)
        rough.Q = 0.0
        rough.H = 0.0
    elif hasattr(snowpack.substrate, "Q") and hasattr(snowpack.substrate, "H"):
        rough.model = 'QH'
        epsr_ground = snowpack.substrate.permittivity_model(sensor.frequency, Tg)
        rough.Q = snowpack.substrate.Q
        rough.H = snowpack.substrate.H
        if hasattr(snowpack.substrate, "N") and snowpack.substrate.N != 2:
            print("Warning: DMRT QMS with QH model assumes N=2. You should set N=2 to avoid this warning.")

    elif hasattr(snowpack.substrate, "roughness_rms"):
        print("Warning: DMRT-QMS does not implement the same version of the Wegmuller & Mazler model")
        rough.model = 'WM'
        epsr_ground = snowpack.substrate.permittivity_model(sensor.frequency, Tg)
        rough.s = snowpack.substrate.roughness_rms

    diameter = np.float64([lay.microstructure.radius * 200 for lay in snowpack.layers])
    density = np.float64([lay.frac_volume * DENSITY_OF_ICE / 1000 for lay in snowpack.layers])
    thickness = np.float64([lay.thickness * 100.0 for lay in snowpack.layers])
    stickiness = np.float64([min(lay.microstructure.stickiness, 1000.0) for lay in snowpack.layers])
    temperature = np.float64([lay.temperature for lay in snowpack.layers])

    TbV, TbH, deg0, ot, albedo, epsr_snow = octave.DMRT_QMS_passive(sensor.frequency / GHz,
                                                                    diameter, density, stickiness,
                                                                    thickness, temperature,
                                                                    Tg, epsr_ground, rough, nout=6)

    # squeeze extra dimension
    deg0 = deg0.squeeze()

    # interpolate
    TbV = np.interp(np.degrees(sensor.theta), deg0, TbV.squeeze())
    TbH = np.interp(np.degrees(sensor.theta), deg0, TbH.squeeze())

    coords = [('theta', sensor.theta), ('polarization', ['V', 'H'])]

    if full_output:
        ke = ot / np.array([lay.thickness for lay in snowpack.layers])
        ks = albedo * ke
        ka = (1 - albedo) * ke
        return PassiveResult(np.vstack([TbV, TbH]).T, coords), ks, ka, epsr_snow
    else:
        return PassiveResult(np.vstack([TbV, TbH]).T, coords)


#
# crash with octave
#
def dmrt_qms_active(sensor, snowpack):

    print("Be careful, the returned results are completely wrong with my octave version.")
    Tg = snowpack.substrate.temperature if snowpack.substrate is not None else 273.0

    ratio = 7.0
    rms = 0.10
    surf_model = 'NMM3D'       # pre-built NMM3D look up table
    epsr_ground = 5.0 + 0.5j

    diameter = np.float64([lay.microstructure.radius * 200 for lay in snowpack.layers])
    density = np.float64([lay.frac_volume * DENSITY_OF_ICE / 1000 for lay in snowpack.layers])
    thickness = np.float64([lay.thickness * 100 for lay in snowpack.layers])
    stickiness = np.float64([min(lay.microstructure.stickiness, 1000) for lay in snowpack.layers])
    temperature = np.float64([lay.temperature for lay in snowpack.layers])

    vv = []
    hh = []
    for deg0inc in np.degrees(sensor.theta_inc):
        res = octave.DMRT_QMS_active(sensor.frequency / GHz, float(deg0inc),
                                     thickness, density, diameter, stickiness, temperature,
                                     epsr_ground, rms, ratio, surf_model, nout=15)
        print(res)
        vvdb, hvdb, vhdb, hhdb, ot, albedo, epsr_eff, vv_vol, hv_vol, vh_vol, hh_vol, vv_surf, hv_surf, vh_surf, hh_surf = res
        vv.append(vvdb)
        hh.append(hhdb)

    return vv, hh


def dmrt_qms_emmodel(sensor, layer, dmrt_qms_path=None):
    """ Compute scattering and absorption coefficients using DMRT QMS

        :param layer: describe the layer.
        :param sensor: describe the sensor configuration.
"""

    diameter = np.float64([layer.microstructure.radius * 200])
    density = np.float64([layer.frac_volume * DENSITY_OF_ICE / 1000])
    thickness = np.float64([layer.thickness * 100.0])
    stickiness = np.float64([min(layer.microstructure.stickiness, 1000.0)])
    temperature = np.float64([layer.temperature])

    ot, albedo, epsr_snow = octave.DMRT_QMS_coefs(sensor.frequency / GHz,
                                                  diameter, density, stickiness,
                                                  thickness, temperature, nout=3)

    ke = ot / layer.thickness
    ks = albedo * ke
    ka = (1 - albedo) * ke

    nt = namedtuple("dmrt_qms_emmodel", "ks ka")
    return nt(ks=ks, ka=ka)
