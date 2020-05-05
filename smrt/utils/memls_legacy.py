# coding: utf-8

""" Wrapper to the original MEMLS matlab code using the SMRT framework. To use this module, extra installation are needed:

 * download MEMLS from http://www.iapmw.unibe.ch/research/projects/snowtools/memls.html. Decompress the archive somewhere on your disk.

 * install the oct2py module using :code:`pip install oct2py` or :code:`easy_install install oct2py` 

 * install Octave version 3.6 or above.

 * for convenience you can set the MEMLS_DIR environment variable to point to MEMLS path. This path can also be programmatically set with :py:func:`set_memls_path`

 In case of problem check the instructions given in http://blink1073.github.io/oct2py/source/installation.html

"""

import os
from tempfile import NamedTemporaryFile
from collections.abc import Sequence
from collections import namedtuple
import itertools

import numpy as np

from oct2py import octave

from smrt.core.result import PassiveResult, ActiveResult, concat_results
from smrt import SMRTError
from smrt.core.sensitivity_study import SensitivityStudy
from smrt.core.globalconstants import DENSITY_OF_ICE

# MEMLS model

ABORN = 12  # we recommend to use ABORN.
MEMLS_RECOMMENDED = 11

# python-space path to memls
_memls_path = None

def set_memls_path(path):
    """set the path where MEMLS archive has been uncompressed, i.e. where the file `memlsmain.m` is located."""
    global _memls_path

    if path != _memls_path:
        #octave.restoredefaultpath() # risk of bad interference with DMRT_QMS
        octave.addpath(path)
        octave.addpath(os.path.dirname(__file__))
        _memls_path = path


try:
    # set
    set_memls_path(os.environ['MEMLS_DIR'])
except KeyError:
    pass


def run(sensor, snowpack, scattering_choice=ABORN, atmosphere=None, memls_path=None, memls_driver=None, snowpack_dimension=None):
    """ call MEMLS for the snowpack and sensor configuration given as argument. Any microstructure model that defines the "corr_length" parameter
        is valid, but it must be clear that MEMLS only considers exponential autocorrelation.

        :param snowpack: describe the snowpack.
        :param sensor: describe the sensor configuration.
        :param scattering_choice: MEMLS proposes several formulation to compute scattering_function. scattering_choice=ABORN (equals 12) is the default
            here and is recommended choice to compare with IBA. Note that some comments in memlsmain.m suggest to use 
            scattering_choice=MEMLS_RECOMMENDED (equals 11). Note also that the default grain type in memlsmain is graintype=1 
            corresponding to oblate spheroidal calculation of effective permittivity from the empirical representation of depolarization factors. To use a Polder-Van Santen representation of effective permittivity for small spheres, graintype=2 must be set in your local copy of MEMLS.
        :param atmosphere: describe the atmosphere. Only tbdown is used for the Tsky argument of memlsmain.
        :param memls_path: directory path to the memls Matlab scripts
        :param memls_driver: matlab function to call to run memls. memlsmain.m is the default driver in the original MEMLS distribution for the passive case and amemlsmain.m for the active case.
        :param snowpack_dimension: name and values (as a tuple) of the dimension to create for the results when a list of snowpack is provided. E.g. time, point, longitude, latitude. By default the dimension is called 'snowpack' and the values are from 1 to the number of snowpacks.


"""

    if memls_path is not None:
        set_memls_path(memls_path)

    if isinstance(sensor.frequency, Sequence) or isinstance(sensor.frequency, np.ndarray):
        raise SMRTError("Sensor must have a single frequency for running memls_legagcy")

    if isinstance(snowpack, SensitivityStudy):
            snowpack_dimension = (snowpack.variable, snowpack.values)
            snowpack = snowpack.snowpacks.tolist()

    if isinstance(snowpack, Sequence):
        result_list = [run(sensor, sp, scattering_choice=scattering_choice,
                           atmosphere=atmosphere, memls_driver=memls_driver) for sp in snowpack]
        if snowpack_dimension is None:
            snowpack_dimension = 'snowpack', range(len(snowpack))
        return concat_results(result_list, snowpack_dimension)

    Tsky = atmosphere.tbdown(sensor.frequency, np.cos(sensor.theta), 1) if atmosphere is not None else 0
    Tgnd = snowpack.substrate.temperature if snowpack.substrate is not None else 273

    if snowpack.substrate is None:
        ground_reflH = itertools.repeat(0)
        ground_reflV = itertools.repeat(0)
    else:
        print("Using MEMLS with substrate has not been tested. Provide feeback if it works (or not)")
        eps_1 = snowpack.layers[-1].permittivity(1, sensor.frequency)
        print("Warning: the permittivity of the ice in the last layer is used instead of the effective permittivity to compute the reflection of the subtrate. This is an approximation that needs to be changed. Please contact developer for any serious simulations with soil...")
        m = snowpack.substrate.specular_reflection_matrix(sensor.frequency, eps_1, np.cos(sensor.theta), 2)
        ground_reflH = m.diagonal()[1::2]
        ground_reflV = m.diagonal()[0::2]

    # prepare the input file in a temporary file
    with NamedTemporaryFile("w", delete=False) as f:
        #     layer-number, temp [K], volume fraction of liquid water, density [kg/m3],
        # thickness [cm], Salinity (0 - 0.1) [ppt], expon.corr.length [mm]

        for ilay, lay in enumerate(reversed(snowpack.layers)):
            f.write("%i, %g, %g, %g, %g, %g, %g\n" % (ilay+1, lay.temperature, lay.liquid_water, lay.frac_volume*DENSITY_OF_ICE, lay.thickness*100.0,
                                                      lay.salinity, lay.microstructure.corr_length*1000.))

    # uncomment these lines if you need to check the input file content.
    #with open(f.name) as ff:
    #    print(ff.readlines())

    if memls_driver is None:
        memls_driver = "memlsmain" if sensor.mode == 'P' else "amemlsmain"

    memlsfct = getattr(octave, memls_driver)

    if sensor.mode == 'P':
        res = [memlsfct(sensor.frequency*1e-9, thetad,
                    float(reflH), float(reflV), f.name, float(Tsky), float(Tgnd), scattering_choice)
                for thetad, reflH, reflV in zip(sensor.theta_deg, ground_reflH, ground_reflV)]
        res = np.vstack(res)
        coords = [('theta', sensor.theta_deg), ('polarization', ['V', 'H'])]

    else: # active
        mean_slope = 1e3  # a high value to remove this contribution. But in the future should be taken from the substrate model, depending on the model...
        res = [memlsfct(sensor.frequency*1e-9, thetad,
                    float(reflH), float(reflV), float(reflH), float(reflV), f.name, float(Tsky), float(Tgnd), scattering_choice, mean_slope, 0)
                    ['sigma0'][0, :]
                for thetad, reflH, reflV in zip(sensor.theta_inc_deg, ground_reflH, ground_reflV)]

        coords = [('polarization', ['V', 'H']), ('polarization_inc', ['V', 'H']), ('theta_inc', sensor.theta_inc_deg), ('theta', sensor.theta_deg)]
        res = np.array(res)
        norm = 4 * np.pi * np.cos(sensor.theta)  # convert back backscattering coefficient
        # assemble in the polarizations
        res = [[np.diagflat(res[:, 0] / norm), np.diagflat(res[:, 2] / norm)], 
               [np.diagflat(res[:, 2] / norm), np.diagflat(res[:, 1]) / norm]]

    os.unlink(f.name)

    if sensor.mode == 'P':
        return PassiveResult(res, coords)
    else: # sensor.mode == 'A':
        return ActiveResult(res, coords)


def memls_emmodel(sensor, layer, scattering_choice=ABORN, graintype=2):
    """ Compute scattering (gs6) and absorption coefficients (gai) using MEMLS

        :param layer: describe the layer.
        :param sensor: describe the sensor configuration.
        :param scattering_choice: MEMLS proposes several formulation to compute scattering_function. scattering_choice=ABORN (equals 12) is the defaut here and is recommended choice to compare with IBA."""

    res = octave.memlsscatt(sensor.frequency/1e9, float(layer.temperature), float(layer.liquid_water), layer.frac_volume*DENSITY_OF_ICE,
                            float(layer.salinity), layer.microstructure.corr_length*1000.0, scattering_choice, graintype)

    nt = namedtuple("memls_emmodel", "ks ka")
    return nt(ks=res[0, 0], ka=res[0, 1])
