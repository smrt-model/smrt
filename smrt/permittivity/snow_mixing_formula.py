
# coding: utf-8

"""Mixing formulae relevant to snow. This module contains equations to compute the effective permittivity of snow.
Many semi-empirical mixing formulae have been developed for specific mixture of two, three or more materials (e.g. snow).

These functions are to be used with py:meth:`~smrt.emmodel.iba.derived_IBA` or
py:meth:`~smrt.emmodel.symsce_torquato21.derived_SymSCETK21` to change the default of most emmodels (IBA, DMRT, SFT
Rayleigh, SCE) using the generic mixing formula Polder van Staten that automatically mixes the permittivities of the
background (e.g.) and the scatterer materials (e.g. ice) to compute the effective permittivity of snow in a proportion
determined by frac_volume.

They should not be used to set the material permittivities as input of py:meth:`~smrt.smrt_inputs.make_snowpack` and
similar functions (because the emmodel would re-mix the already mixed materials with the background material).
"""

import numpy as np
from ..core.layer import layer_properties
from ..core.globalconstants import FREEZING_POINT, DENSITY_OF_ICE, DENSITY_OF_WATER
from ..core.error import SMRTError, smrt_warn
from .generic_mixing_formula import polder_van_santen, polder_van_santen_three_components, polder_van_santen_three_spherical_components


@layer_properties("temperature", "density", "liquid_water", optional_arguments=["ice_permittivity_model", "water_permittivity_model"])
def wetsnow_permittivity_tinga73(frequency, temperature, density, liquid_water, ice_permittivity_model=None, water_permittivity_model=None):
    """Computes the effective permittivity proposed by Tinga et al. 1973 for three-component mixing. The component 1 is the background ("a" here),
    the compoment 2 ("w" here) is a spherical shell surrounding the component 3 ("i" here).

     It was used by Tiuri as well as T. Mote to compute wet snolw permittivity.

    Tinga, W.R., Voss, W.A.G. and Blossey, D. F.: General approach to multiphase dielectric mixture theory.
    Journal of Applied Physics, Vol.44(1973) No.9,pp.3897-3902.
    doi: /10.1063/1.1662868

    Tiuri, M. and Schultz, H., Theoretical and experimental studies of microwave radiation from a natural snow field. In Rango, A. , ed.
    Microwave remote sensing of snowpack properties. Proceedings of a workshop ... Fort Collins, Colorado, May 20-22, 1980.
    Washington, DC, National Aeronautics and Space Center, 225-234. (Conference Publication 2153.)

"""

    if (temperature < FREEZING_POINT) and np.any(liquid_water > 0):
        raise SMRTError("Liquid water is positive and temperature is set below freezing. This seems incompatible.")

    # wetness W is the weight percentage of liquid water contained in the snow
    W = liquid_water * DENSITY_OF_WATER / (liquid_water * DENSITY_OF_WATER + (1 - liquid_water) * DENSITY_OF_ICE)

    # equation for spheres. Here we rather defined V to avoid the exponentiation
    # ri = 0.5e-3  # the result is independent on this value, because only ratio rw/ri or ra/ri or rw/ra are used

    # rw = ri * (1 + DENSITY_OF_ICE / DENSITY_OF_WATER * W / (1 - W))**(1 / 3)

    # ra = ri * ((DENSITY_OF_ICE / density) * (1 + W / (1 - W)))**(1 / 3)

    Vw_i = 1 + DENSITY_OF_ICE / DENSITY_OF_WATER * W / (1 - W)
    Va_i = (DENSITY_OF_ICE / density) * (1 + W / (1 - W))

    if water_permittivity_model is None:
        from .water import water_permittivity_tiuri80
        water_permittivity_model = water_permittivity_tiuri80
    if ice_permittivity_model is None:
        from .ice import ice_permittivity_tiuri84
        ice_permittivity_model = ice_permittivity_tiuri84

    eps_a = 1
    eps_w = water_permittivity_model(frequency, temperature=FREEZING_POINT)
    eps_i = ice_permittivity_model(frequency, temperature=temperature)  # this must be dry ice !

    alpha = 2 * eps_w + eps_i
    diff_wi = eps_w - eps_i
    diff_wa = eps_w - eps_a

    denominator = (2 * eps_a + eps_w) * alpha - 2 * (1 / Vw_i) * diff_wa * diff_wi \
        - (Vw_i / Va_i) * diff_wa * alpha \
        + (1 / Va_i) * diff_wi * (2 * eps_w + eps_a)

    Es = eps_a * (1 + 3 * ((Vw_i / Va_i) * diff_wa * alpha - (1 / Va_i) * diff_wi * (2 * eps_w + eps_a)) / denominator)

    # t is possible to compute the square_field_ratio_tinga73
    # using np.abs(eps_w * eps_a / denominator)**2
    # to be implemented

    return Es


def compute_frac_volumes(density, liquid_water):
    """Computes the fractional volume of ice+water, the fractional volume of ice, and the fractional volume of water 
    from the (wet) snow density and the liquid_water which is the volume fraction of liquid with respect to ice + liquid (but no air).

    :param density: density of the snow, including the ice and water phases.
    :param liquid_water: (fractional volume of water with respect to ice+water volume).

    :returns: frac_volume, fi, fw
"""
    density_melange = DENSITY_OF_ICE * (1 - liquid_water) + DENSITY_OF_WATER * liquid_water
    # variations of density with temperature (a few kg/m3) and air mass (less than 1 kg/m3) are not taken into account.

    frac_volume = density / density_melange  #

    fi = frac_volume * (1 - liquid_water)

    fw = frac_volume * liquid_water

    return frac_volume, fi, fw


@layer_properties("temperature", "density", "liquid_water", optional_arguments=["ice_permittivity_model", "water_permittivity_model"])
def wetsnow_permittivity_colbeck80_caseI(frequency, temperature, density, liquid_water, ice_permittivity_model=None, water_permittivity_model=None):
    """Computes the effective permittivity proposed by Colbeck, 1980 for the pendular regime.

    Colbeck, S. C. (1980). Liquid distribution and the dielectric constant of wet snow.
    Goddard Space Flight Center Microwave Remote Sensing of Snowpack Properties, 21–40.

"""
    if (temperature < FREEZING_POINT) and np.any(liquid_water > 0):
        raise SMRTError("Liquid water is positive and temperature is set below freezing. This seems incompatible.")

    ice_permittivity_model, water_permittivity_model = default_ice_water_permittivity(ice_permittivity_model, water_permittivity_model)

    Ac = 0.422  # page 24
    Asnow = [(1 - Ac) / 2, (1 - Ac) / 2, 0.422]

    # for n = 3.5 (page 4), we read in Fig 2 page 31:
    m = 0.072  # this value is different from Löwe et al. TC (2013) and I don't know why.
    Ac = 1 / (1 + 2 / m)

    Awater = [(1 - Ac) / 2, (1 - Ac) / 2, Ac]

    frac_volume, fi, fw = compute_frac_volumes(density, liquid_water)

    return polder_van_santen_three_components(
        f1=fi,
        f2=fw,
        eps0=1,
        eps1=ice_permittivity_model(frequency, temperature=temperature),
        eps2=water_permittivity_model(frequency, temperature=FREEZING_POINT),
        A1=Asnow,
        A2=Awater
    )


@layer_properties("temperature", "density", "liquid_water", optional_arguments=["ice_permittivity_model", "water_permittivity_model"])
def wetsnow_permittivity_colbeck80_caseII(frequency, temperature, density, liquid_water, ice_permittivity_model=None, water_permittivity_model=None):
    """Computes the effective permittivity proposed by Colbeck, 1980 for the funicular regime and low dry snow density.

    Colbeck, S. C. (1980). Liquid distribution and the dielectric constant of wet snow.
    Goddard Space Flight Center Microwave Remote Sensing of Snowpack Properties, 21–40.

"""

    if (temperature < FREEZING_POINT) and np.any(liquid_water > 0):
        raise SMRTError("Liquid water is positive and temperature is set below freezing. This seems incompatible.")

    ice_permittivity_model, water_permittivity_model = default_ice_water_permittivity(ice_permittivity_model, water_permittivity_model)

    frac_volume, fi, fw = compute_frac_volumes(density, liquid_water)

    return polder_van_santen_three_spherical_components(
        f1=fi,               # ice fractional volume
        f2=1 - frac_volume,  # air fractional volume
        eps0=water_permittivity_model(frequency, temperature=FREEZING_POINT),
        eps1=ice_permittivity_model(frequency, temperature=temperature),
        eps2=1,
    )

@layer_properties("temperature", "density", "liquid_water", optional_arguments=["ice_permittivity_model", "water_permittivity_model"])
def wetsnow_permittivity_colbeck80_caseIII(frequency, temperature, density, liquid_water, ice_permittivity_model=None, water_permittivity_model=None):
    """Computes the effective permittivity proposed by Colbeck, 1980 for the low porosity.

    Colbeck, S. C. (1980). Liquid distribution and the dielectric constant of wet snow.
    Goddard Space Flight Center Microwave Remote Sensing of Snowpack Properties, 21–40.

"""
    if (temperature < FREEZING_POINT) and np.any(liquid_water > 0):
        raise SMRTError("Liquid water is positive and temperature is set below freezing. This seems incompatible.")

    ice_permittivity_model, water_permittivity_model = default_ice_water_permittivity(ice_permittivity_model, water_permittivity_model)

    # for n = 3.5 (page 26), we read in Fig 2 page 31:
    m = 0.072  
    Ac = 1 / (1 + 2 / m)

    Awater = [(1 - Ac) / 2, (1 - Ac) / 2, Ac]


    Ac = 1 / 3  # air bubble
    Aair = [Ac, Ac, Ac]

    frac_volume, fi, fw = compute_frac_volumes(density, liquid_water)

    return polder_van_santen_three_components(
        f1=fw,
        f2=1 - frac_volume,
        eps0=ice_permittivity_model(frequency, temperature=temperature),
        eps1=water_permittivity_model(frequency, temperature=FREEZING_POINT),
        eps2=1,
        A1=Awater,
        A2=Aair
    )

@layer_properties("density", "liquid_water")
def wetsnow_permittivity_hallikainen86(frequency, density, liquid_water):
    """Computes the effective permittivity of a snow mixture calculated with the Modified Debye model by Hallikainen 1986

    The implemented equation are 10, 11 and 13a-c.

    The validity of the model is: frequency between 3 and 37GHz;
                                  mv between 1% and 12%;
                                  dry_snow_density between 0.09 and 0.38g/cm3.

    The implementation of this function follows the equations formulation of the original paper 
    Hallikainen, M., F. Ulaby, and M. Abdelrazik, “Dielectric properties of snow in 3 to 37 GHz range,”
    IEEE Trans. on Antennasand Propagation,Vol. 34, No. 11, 1329–1340, 1986. DOI: 10.1109/TAP.1986.1143757
    Anyway this formulation does not allow the reproduction of the results as reported in the paper.
    A new formulation of eq. 12a have been presented in the book
    Microwave Radar and Radiometric Remote Sensing by Ulaby et al. 2014 from which the SMRT function 
    wetsnow_permittivity_hallikainen86_ulaby14 have been implemented. The users are pointed to that definition.


    """

    smrt_warn("This model cannot reproduce the results of the original paper. You may want to use wetsnow_permittivity_hallikainen86_ulaby14 instead.")

    frac_volume, fi, fw = compute_frac_volumes(density, liquid_water)

    # fractional volume of water in %
    mv = 100 * fw

    # Eq 3 in H86 defines the dry snow by (snow density - mass of water per volume of snow) / (1 - volume fo water per volume of snow)
    dry_snow_density_gcm3 = 1e-3 * (density - DENSITY_OF_WATER * fw) / (1 - fw)

    freqGHz = frequency * 1e-9

    # Eq 13
    A1 = 0.78 + 0.03 * freqGHz - 0.58e-3 * freqGHz**2
    A2 = 0.97 - 0.39e-2 * freqGHz + 0.39e-3 * freqGHz**2
    B1 = 0.31 - 0.05 * freqGHz + 0.87e-3 * freqGHz**2

    # Eq 12 (different from Ulaby14)
    A = 1 + 1.83 * dry_snow_density_gcm3 + 0.02 * A1 * mv**1.015 + B1
    B = 0.073 * A1
    C = 0.073 * A2
    x = 1.31

    freq0 = 9.07  # GHz

    eps_ws_r = A + B * mv**x / (1 + (freqGHz / freq0)**2)

    eps_ws_i = C * mv**x * (freqGHz / freq0) / (1 + (freqGHz / freq0)**2)

    return eps_ws_r + 1j * eps_ws_i


@layer_properties("density", "liquid_water")
def wetsnow_permittivity_hallikainen86_ulaby14(frequency, density, liquid_water):
    """Computes the effective permittivity of a snow mixture calculated with the Modified Debye model by Hallikainen 1986
    and revised in Microwave Radar and Radiometric Remote Sensing by Ulaby et al. 2014 
    Equations implemented are ch 4 pp 143-15 4.60a - 4.61h.

    The validity of the model is: frequency between 3 and 37GHz;
                                  mv between 1% and 12%;
                                  dry_snow_density between 0.09 and 0.38g/cm3.

    Same formulation can be reproduced by the book code https://mrs.eecs.umich.edu/codes/Module4_6/Module4_6.html
    """

    frac_volume, fi, fw = compute_frac_volumes(density, liquid_water)

    # fractional volume of water in %
    mv = 100 * fw

    # Eq 3 in H86 defines the dry snow by (snow density - mass of water per volume of snow) / (1 - volume fo water per volume of snow)
    dry_snow_density_gcm3 = 1e-3 * (density - DENSITY_OF_WATER * fw) / (1 - fw)

    freqGHz = frequency * 1e-9

    if np.any(mv > 12) \
            or np.any(dry_snow_density_gcm3 < 0.09) or np.any(dry_snow_density_gcm3 > 0.38) \
            or np.any(freqGHz < 3) or np.any(freqGHz > 37):
        smrt_warn("Hallikainen86_ulaby14 is only valid for mv < 12  dry_snow_density in [0.09...0.38] and freq in [3...37 GHz]")

    # Eq 4.61 f - h
    A1 = 0.78 + 0.03 * freqGHz - 0.58e-3 * freqGHz**2
    A2 = 0.97 - 0.39e-2 * freqGHz + 0.39e-3 * freqGHz**2
    B1 = 0.31 - 0.05 * freqGHz + 0.87e-3 * freqGHz**2   # <-- this leads to a too low epsilon compared to the H86 graphs

    # Eq 4.61 a - e
    A = A1 * (1.0 + 1.83 * dry_snow_density_gcm3 + 0.02 * mv**1.015) + B1
    B = 0.073 * A1
    C = 0.073 * A2
    x = 1.31
    freq0 = 9.07  # GHz

    # Eq 4.60 a - b
    eps_ws_r = A + B * mv**x / (1 + (freqGHz / freq0)**2)
    eps_ws_i = C * mv**x * (freqGHz / freq0) / (1 + (freqGHz / freq0)**2)

    return eps_ws_r + 1j * eps_ws_i


@layer_properties("temperature", "density", "liquid_water", optional_arguments=["ice_permittivity_model"])
def wetsnow_permittivity_wiesmann99(frequency, temperature, density, liquid_water, ice_permittivity_model=None):
    """Computes the effective permittivity of a snow mixture as presented in MEMLS by Wiesmann and Matzler, 1999. Note that the version implemented
    in MEMLS v3 is different.

"""
    if (temperature < FREEZING_POINT) and np.any(liquid_water > 0):
        raise SMRTError("Liquid water is positive and temperature is set below freezing. This seems incompatible.")

    if ice_permittivity_model is None:
        from .ice import ice_permittivity_maetzler06
        ice_permittivity_model = ice_permittivity_maetzler06

    frac_volume, fi, Wi = compute_frac_volumes(density, liquid_water)

    eps_dry = polder_van_santen(fi, e0=1, eps=ice_permittivity_model(frequency, temperature=temperature))  # permittivity of dry snow

    Aa = 0.005    # depolarisation factors of prolate
    Ab = 0.4975   # water inclusion (Matzler 1987)
    Ac = Ab

    eps_sw = 88
    eps_inf_w = 4.9
    f0w = 9e9  # GHz

    eps_eff = 0

    for Ak in [Aa, Ab, Ac]:
        eps_s_k = Wi / 3 * (eps_sw - eps_dry) / (1 + Ak * (eps_sw / eps_dry - 1))
        eps_inf_k = Wi / 3 * (eps_inf_w - eps_dry) / (1 + Ak * (eps_inf_w / eps_dry - 1))
        f0_k = f0w * (1 + Ak * (eps_sw - eps_inf_w) / (eps_dry + Ak * (eps_inf_w - eps_dry)))

        eps_k = eps_inf_k + (eps_s_k - eps_inf_k) / (1 - 1j * frequency / f0_k)

        eps_eff += eps_k

    return eps_dry + eps_eff


@layer_properties("temperature", "density", "liquid_water", optional_arguments=["ice_permittivity_model", "water_permittivity_model"])
def wetsnow_permittivity_memls(frequency, temperature, density, liquid_water, ice_permittivity_model=None, water_permittivity_model=None):
    """Computes the effective permittivity of a snow mixture as calculated in MEMLS using Maxwell-Garnett Mixing rule of water in dry snow
    for prolate spheroidal water with experimentally determined. Dry snow permittivity is here determined with Polder van Santen.

"""
    # %   depolarisation factors.
    # %   calculates complex dielectric constant of wet snow
    # %   using Maxwell-Garnett Mixing rule of water in dry snow
    # %   for prolate spheroidal water with experimentally determined
    # %   depolarisation factors.
    # %   Water temperature is at 273.15 K, with epsilon
    # %   of water from Liebe et al. 1991.
    # %       epsd:  complex epsilon of dry snow
    # %       f:   frequency [GHz]
    # %       Ti:  physical snow temperature [K]
    # %       Wi:  wetness [volume fraction]
    # %
    # %   Version history:
    # %      1.0    wi 15.7.95
    # %      2.0    ma 31.5.2005: Wi is volume fraction (not %)
    # %      3.0    ma 2.4.2007 : adjustments, new function name
    # %   Uses: epswater (since Version 3)
    # %
    # %   Copyright (c) 1997 by the Institute of Applied Physics,
    # %   University of Bern, Switzerland
    # %

    if (temperature < FREEZING_POINT) and np.any(liquid_water > 0):
        raise SMRTError("Liquid water is positive and temperature is set below freezing. This seems incompatible.")

    ice_permittivity_model, water_permittivity_model = default_ice_water_permittivity(ice_permittivity_model, water_permittivity_model)

    Aa = 0.005    # depolarisation factors of prolate
    Ab = 0.4975   # water inclusion (Matzler 1987)
    # Ac = Ab

    ew = water_permittivity_model(frequency, temperature=FREEZING_POINT)

    frac_volume, fi, Wi = compute_frac_volumes(density, liquid_water)

    epsd = polder_van_santen(np.clip(fi, 0, 1),
                             e0=1, eps=ice_permittivity_model(frequency, temperature=temperature))  # permittivity of dry snow

    Ka = epsd / (epsd + Aa * (ew - epsd))
    Kb = epsd / (epsd + Ab * (ew - epsd))
    K = (Ka + 2 * Kb) / 3
    epsz = (1 - Wi) * epsd + Wi * ew * K
    epsn = 1 - Wi * (1 - K)
    eps = epsz / epsn  # Maxwell-Garnett Mixing of water in dry snow
    return eps


@layer_properties("temperature", "density", "liquid_water", optional_arguments=["ice_permittivity_model", "water_permittivity_model"])
def wetsnow_permittivity_three_component_polder_van_santen(frequency, temperature, density, liquid_water,
                                                           ice_permittivity_model=None, water_permittivity_model=None):
    """Computes the effective permittivity of a snow mixture using the three components polder_van_santen, assuming spherical inclusions

"""
    if (temperature < FREEZING_POINT) and np.any(liquid_water > 0):
        raise SMRTError("Liquid water is positive and temperature is set below freezing. This seems incompatible.")

    ice_permittivity_model, water_permittivity_model = default_ice_water_permittivity(ice_permittivity_model, water_permittivity_model)

    if (np.array(density).ndim >= 1) or (np.array(liquid_water).ndim >= 1):
        # density = np.array(density)
        # liquid_water = np.array(liquid_water)
        def func(dens, liq):
            return wetsnow_permittivity_three_component_polder_van_santen(frequency, temperature, dens, liq,
                                                                          ice_permittivity_model=ice_permittivity_model,
                                                                          water_permittivity_model=water_permittivity_model)

        return np.vectorize(func)(density, liquid_water)

    frac_volume, fi, fw = compute_frac_volumes(float(density), liquid_water)

    return polder_van_santen_three_spherical_components(fi, fw,
                                                        eps0=1,
                                                        eps1=ice_permittivity_model(frequency, temperature=temperature),
                                                        eps2=water_permittivity_model(frequency, temperature=FREEZING_POINT))


@ layer_properties("density")
def depolarization_factors_maetzler96(density):
    """The empirical depolarization factors of snow estimated by Mäzler 1996. It is supposed to provide more accurate
    permittivity=f(density) than using constant depolarization factors in Polder van Santen (e.g. spheres)

Biblio: C. Mäzler, Microwave Permittivity of dry snow, IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 34, NO. 2, MARCH 1996
"""
    frac_volume = density / DENSITY_OF_ICE   # this way to compute frac_volume avoid inversion of the medium. For dry snow only

    if frac_volume < 0.33:
        A = 0.1 + 0.5 * frac_volume
    elif frac_volume < 0.71:
        A = 0.18 + 3.24 * (frac_volume - 0.49)**2
    else:
        A = 1 / 3
    return np.array([A, A, 1 - 2 * A])


@ layer_properties("density")
def drysnow_permittivity_maetzler96(density, e0=1, eps=3.185):

    if (e0.real > 1) and (eps == 1):
        e0, eps = eps, e0

    assert e0.real < eps.real

    frac_volume = density / DENSITY_OF_ICE   # this way to compute frac_volume avoid inversion of the medium

    A = depolarization_factors_maetzler96(density)

    # A = np.array([1 / 3., 1 / 3., 1 / 3.]) # Spheres. For testing

    eps_diff = eps - e0

    # Solve Polder van Santen with an iterative approach (could be optimized)
    # rough first guess
    eps_eff0 = frac_volume * eps + (1 - frac_volume) * e0

    for i in range(20):  # use an inefficient iterative approach
        eps_app = e0 * A + eps_eff0 * (1 - A)

        eps_eff = e0 + frac_volume * eps_diff * np.sum(eps_app / (eps_app + A * eps_diff)) \
            / (3 - frac_volume * eps_diff * np.sum(A / (eps_app + A * eps_diff)))

        if np.abs(eps_eff - eps_eff0) < 1e-6:
            break
        eps_eff0 = eps_eff  # new estimate becomes first guess

    # last estimation of eps_app: eps_app = e0 + (1 - A) * eps_eff

    return eps_eff


def default_ice_water_permittivity(ice_permittivity_model, water_permittivity_model):

    if ice_permittivity_model is None:
        from .ice import ice_permittivity_maetzler06
        ice_permittivity_model = ice_permittivity_maetzler06

    if water_permittivity_model is None:
        from .water import water_permittivity_maetzler87
        water_permittivity_model = water_permittivity_maetzler87

    return ice_permittivity_model, water_permittivity_model
