# coding: utf-8

"""This module provides a function to build soil model and provides some soil permittivity formulae.

To create a substrate, use/implement an helper function such as :py:func:`~smrt.substrate.substrate.make_soil`. This function is able to 
automatically load a specific soil model and provides some soil permittivity formulae as well.

Examples::

    from smrt import make_soil
    soil = make_soil("soil_wegmuller", "dobson85", moisture=0.2, sand=0.4, clay=0.3, drymatter=1100, roughness_rms=1e-2)

It is recommand to first read the documentation of :py:func:`~smrt.substrate.substrate.make_soil` and then explore the different types of soil
models.

"""

from functools import partial
from numbers import Number

import numpy as np
import scipy


# local import
from smrt.core.error import SMRTError
from smrt.core.interface import get_substrate_model
from smrt.core.globalconstants import PERMITTIVITY_OF_FREE_SPACE


def make_soil(substrate_model, permittivity_model, temperature, moisture=None,
              sand=None, clay=None, drymatter=None, **kwargs):
    """ Construct a soil instance based on a given surface electromagnetic model, a permittivity model and parameters 

    :param substrate_model: name of substrate model, can be a class or a string. e.g. fresnel, wegmuller...
    :param permittivity_model: permittivity_model to use. Can be a name ("hut_epss", "dobson85", "montpetit2008"), a function of
        frequency and temperature or a complex value.
    :param moisture: soil moisture in m:sup:`3` m:sup:`-3` to compute the permittivity. This parameter is used depending on the permittivity_model.
    :param sand: soil relative sand content. This parameter is used or not depending on the permittivity_model.
    :param clay: soil relative clay content. This parameter is used or not depending on the permittivity_model.
    :param drymatter: soil content in dry matter in kg m:sup:`-3`. This parameter is used or not depending on the permittivity_model.

    :param **kwargs: geometrical parameters depending on the substrate_model. Refer to the document of each model to see the
        list of required and optional parameters. Usually, it is roughness_rms, corr_length, ...

    **Usage example:**

    ::
        TOTEST: bottom = substrate.make('Flat', permittivity_model=complex('6-0.5j'))
        TOTEST:  bottom = substrate.make('Wegmuller', permittivity_model='soil', roughness_rms=0.25, moisture=0.25)

    """

    # process the permittivity_model argument
    if isinstance(permittivity_model, str):

        if permittivity_model == "hut_epss":
            # return soil_dielectric_constant_hut after setting the parameters
            if moisture is None or sand is None or clay is None or drymatter is None:
                raise SMRTError("The parameters moisture, sand, clay and drymatter must be set")

            permittivity_model = partial(soil_dielectric_constant_hut, SM=moisture, sand=sand, clay=clay, dm_rho=drymatter)

        elif permittivity_model == "dobson85":
            # return soil_dielectric_constant_dobson after setting the parameters
            if moisture is None or sand is None or clay is None:
                raise SMRTError("The parameters moisture, sand, clay must be set")
            permittivity_model = partial(soil_dielectric_constant_dobson, SM=moisture, S=sand, C=clay)

        elif permittivity_model == "montpetit2008":
            permittivity_model = soil_dielectric_constant_monpetit2008

        else:
            raise SMRTError("The permittivity model '%s' is not recongized" % permittivity_model)
    else:
        if isinstance(permittivity_model, Number):  # a constant value
            # create a function with 2 args that always return the same value
            def permittivity_model(frequency, temperature, cst=permittivity_model):
                return cst
        elif not callable(permittivity_model):
            raise SMRTError("The permittivity_model argument is not of the accepted types."
                            "It must be a string with an implemented permittivity model name,"
                            " a number or a function with two arguments.")
        # check that other parameters are
        if moisture is not None or sand is not None or clay is not None or drymatter is not None:
            raise Warning("Setting moisture, clay, sand or drymatter when permittivity_model is a number or function is useless")

    # process the substrate_model argument
    if not isinstance(substrate_model, type):
        substrate_model = get_substrate_model(substrate_model)

    # create the instance
    return substrate_model(temperature, permittivity_model, **kwargs)


# !Dobson et al.,(1985) dielectric constant calculation
# !(extracted from HUTnlayer code [Lemmetyinen et al.,(2010)])
# !(extracted from DMRTML)

def soil_dielectric_constant_dobson(frequency, tempK, SM, S, C):

    e_0 = PERMITTIVITY_OF_FREE_SPACE
    e_w_inf = 4.9
    e_s = 4.7
    rho_b = 1.3
    rho_s = 2.664

    temp = tempK - 273.15

    beta1 = 1.2748 - 0.519 * S - 0.152 * C
    beta2 = 1.33797 - 0.603 * S - 0.166 * C

    sigma_eff = 0.0467 + 0.2204 * rho_b - 0.4111 * S + 0.6614 * C

    e_w0 = 87.134 - 1.949e-1 * temp - 1.276e-2 * temp**2 + 2.491e-4 * temp**3
    rt_w = (1.1109e-10 - 3.824e-12 * temp + 6.938e-14 * temp**2 - 5.096e-16 * temp**3) / (2 * np.pi)

    e_fw1 = e_w_inf + (e_w0 - e_w_inf) / (1 + (2 * np.pi * frequency * rt_w)**2)
    e_fw2 = 2 * np.pi * frequency * rt_w * (e_w0 - e_w_inf) / (1 + (2 * np.pi * frequency * rt_w)**2) + \
        sigma_eff * (rho_s - rho_b) / (2 * np.pi * frequency * e_0 * rho_s * SM)

    return complex((1 + (rho_b / rho_s) * (e_s**0.65 - 1) + SM**beta1 * e_fw1**0.65 - SM)**(1 / 0.65),
                   (SM**beta2 * e_fw2**0.65)**(1 / 0.65))


#! (after Pulliainen et al.1999)
#! extracted from DMRTML

def soil_dielectric_constant_hut(frequency, tempK, SM, sand, clay, dm_rho):

    # Parameters for soil dielectric constant calculation with water
    ew_inf = 4.9

    tempC = tempK - 273.15

    if tempC > 0:  # liquid water
        # calculates real and imag. part of water dielectricity (code HUT 20.12.95 [epsw.m]; K.Tigerstedt)
        ew0 = 87.74 - 0.40008 * tempC + 9.398e-4 * tempC**2 + 1.410e-6 * tempC**3
        d = 25 - tempC
        alfa = 2.033e-2 + 1.266e-4 * d + 2.464e-6 * d**2
        tw = 1 / (2 * np.pi) * (1.1109e-10 - 3.824e-12 * tempC + 6.938e-14 * tempC**2 - 5.096e-16 * tempC**3)

        ew_r = ew_inf + (ew0 - ew_inf) / (1 + (2 * np.pi * frequency * tw)**2)
        ew_i = (ew0 - ew_inf) * 2 * np.pi * frequency * tw / (1 + (2 * np.pi * frequency * tw)**2)
    else:
        raise NotImplementedError("not implemented")
#      !option for salt consideration (Mätzler 1987)
#      !iei_S =A/M+B*M**C                 !impure ice
#      !iei_P=Ap/M+Bp*M**Cp                 !pure ice
#      !delta_iei = iei_S - iei_P
#      !ew_i=ew_i+delta_iei*SS/13

    beta = 1.09 - 0.11 * sand + 0.18 * clay
    epsalf = 1 + 0.65 * dm_rho / 1000.0 + SM**beta * (complex(ew_r, ew_i)**0.65 - 1)  # dm_rho is now in SI // Ulaby et al. (1986, p. 2099)

    return (epsalf)**(1 / 0.65)


def soil_dielectric_constant_monpetit2008(frequency, temperature):
    """Soil dielectric constant formulation based on the formulation Montpetit et al. 2018. 
    The formulation is only valid for below-frrezing point temperature.

    Reference: Montpetit, B., Royer, A., Roy, A., & Langlois, A. (2018). In-situ passive microwave emission model 
    parameterization of sub-arctic frozen organic soils. Remote Sensing of Environment, 205, 112–118. 
    https://doi.org/10.1016/j.rse.2017.10.033

"""
    # from functools import partial
    # from smrt.inputs.make_soil import soil_dielectric_constant_dobson
    if temperature > 273.15:
        raise SMRTError("soil_dielectric_constant_monpetit is not implemented for above freezing temperatures.")
        # moisture=0.2
        # sand=0.4
        # clay=0.3
        # return partial(soil_dielectric_constant_dobson, SM=moisture, S=sand, C=clay)
    # else:

    p = scipy.interpolate.interp1d([10.65e9, 19e9, 37e9],
                                   [complex(3.18, 0.0061), complex(3.42, 0.0051), complex(4.47, 0.33)],
                                   fill_value="extrapolate")
    return p(frequency)
