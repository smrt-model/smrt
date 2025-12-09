# coding: utf-8

"""
Provide the Multi-Fresnel Thermal Emission (MFTE) RT solver for passive sensor.

Multi-Fresnel Thermal Emission (MFTE) is a fast RT solver suitable for passive microwave and none scattering
media. It computes the thermal emission of a multi-layer stack of homogeneous layers (absorption only, no scattering)
with flat interfaces (no roughness) solely characterized by their permittivity and temperature. It is most suitable for
instance for L-band and lower frequencies over the dry zone of the ice-sheet where the penetration is deep and the
stratification of the snowpack and the profile of temperature are crucial to compute the emission.
Note that the layers are incoherent, layer thickness smaller than the wavelength are not recommended (at least not smaller than a
quarter of the wavelength). In principle, MFTE gives the same results as DORT, when the aforementioned assumption are
respected, but much more rapidely.

Even with a small number of layers, MFTE is x30 faster than DORT, and requires much less memory.
The performance difference increases with the number of layers.

The model is inspired from:

Hébert, M., Simonot, L., & Mazauric, S. (2015). Matrix method to predict the spectral reflectance of stratified surfaces
including thick layers and thin films. https://hal.science/hal-01155614

The formulation (with typos) is in the Annex:

P. Zeiger, G. Picard, P. Richaume, A. Mialon, Nemesio Rodriguez-Fernandez. Resolution enhancement of SMOS brightness
temperatures: application to melt detection on the Antarctic and Greenland ice sheets, Remote Sensing of Environment,
315, 114469, http://doi.org/10.1016/j.rse.2024.114469, 2024


**Usage**::

    # Create a model using a nonscattering medium and the rtsolver 'multifresnel_thermalemission'.
    m = make_model("nonscattering", "multifresnel_thermalemission")

"""

import numpy as np
import xarray as xr

# local import
from smrt.core.error import SMRTError, smrt_warn
from smrt.core.result import make_result
from smrt.rtsolver.rtsolver_utils import prepare_kskaeps_profile_information

from .multifresnel.multifresnel import compute_emerging_radiation
from .multifresnel import multifresnel_derivatives


class MultiFresnelThermalEmissionDerivatives(object):
    """
    Implement the Multi-Fresnel Thermal Emission (MFTE) solver for SMRT.

    Args:
        error_handling: If set to "exception" (the default), raise an exception in cause of error, stopping the code.
            If set to "nan", return a nan, so the calculation can continue, but the result is of course unusuable and
            the error message is not accessible. This is only recommended for long simulations that sometimes produce an error.
        prune_deep_snowpack: this value is the optical depth from which the layers are discarded in the calculation.
            This prevents numerical unstability inherent to the MFTE formulation for a very deep snowpack.
            A value of 10 is used by default which is already very large / safe. In case of problems of stability, this
            value should be decreased. Set to None to deactivate pruning, but this is not recommended.
    """

    # this specifies which dimension this solver is able to deal with. Those not in this list must be managed by the called (Model object)
    # e.g. here, frequency, time, ... are not managed
    _broadcast_capability = {"theta", "polarization"}

    def __init__(self, error_handling: str = "exception", prune_deep_snowpack: int = 10):
        self.error_handling = error_handling
        # self.process_coherent_layers = process_coherent_layers
        self.prune_deep_snowpack = prune_deep_snowpack

    def solve(self, snowpack, emmodels, sensor, atmosphere=None):
        """
        Solve the radiative transfer equation for a given snowpack, emmodels and sensor configuration.

        Args:
            snowpack: Snowpack object.
            emmodels: List of electromagnetic models object. With MFTE it is recommended to use
                the nonscattering emmodel even though other emmodels will work with their scattering not used.
            sensor: Sensor object.
            atmosphere: [Optional] Atmosphere object.

        Returns:
            result: Result object.
        """
        if sensor.mode != "P":
            raise SMRTError(
                "the MFTE solver is only suitable for passive microwave. Use an adequate sensor falling in"
                "this catergory."
            )

        if atmosphere is not None:
            raise SMRTError(
                "the MFTE solver can not handle atmosphere yet. Please put an issue on github if thisfeature is needed."
            )

        # for em in emmodels:
        #     if getattr(em, "ks", 0) > 0:
        #         smrt_warn(
        #             "the MFTE solver does not take into account scattering. Use the DORT solver if scattering"
        #             " is significant."
        #         )

        thickness = snowpack.layer_thicknesses
        temperature = snowpack.profile("temperature")
        effective_permittivity = [emmodel.effective_permittivity() for emmodel in emmodels]

        if snowpack.substrate is not None:
            effective_permittivity.append(snowpack.substrate.permittivity(sensor.frequency))
            if effective_permittivity[-1].imag < 1e-8:
                smrt_warn("the permittivity of the substrate has a too small imaginary part for reliable results")
            thickness.append(1e10)  # add an infinite layer (hugly hack)
            temperature = np.append(temperature, snowpack.substrate.temperature)

        mu = np.cos(sensor.theta)
        # dM is not implemented yet
        M, dM, tau_snowpack = multifresnel_derivatives.compute_matrix_slab_derivatives(
            frequency=sensor.frequency,
            outmu=mu,
            permittivity=effective_permittivity,
            temperature=temperature,
            thickness=thickness,
            prune_deep_snowpack=self.prune_deep_snowpack,
        )

        if tau_snowpack < 5 and snowpack.substrate is None:
            smrt_warn(
                "Multifresnel has detected that the snowpack is optically shallow (tau=%g) and no substrate has been set, meaning that the space "
                "under the snowpack is 'empty' with snowpack shallow enough to affect the measured signal at the surface."
                "This is usually not wanted and can produce wrong results. Either increase the thickness of the snowpack or set a substrate."
                " If wanted, add a transparent substrate to supress this warning" % tau_snowpack
            )

        Tbv, Tbh = compute_emerging_radiation(M)

        #  describe the results list of (dimension name, dimension array of value)
        coords = [("theta", sensor.theta_deg), ("polarization", ["V", "H"])]

        # store other diagnostic information
        other_data = prepare_kskaeps_profile_information(
            snowpack, emmodels, effective_permittivity[0 : snowpack.nlayer], mu=mu
        )
        # This could be added to prepare_kskaeps_profile_information
        # compute_emerging_radiation is not implemented yet for derivatives
        dtbsdti = [compute_emerging_radiation(dm) for dm in dM]
        derivative_array_v = [rad[0] for rad in dtbsdti]
        derivative_array_h = [rad[1] for rad in dtbsdti]
        layer_index = "layer", range(len(effective_permittivity))
        polarization_index = "polarization", range(2)
        other_data["dTBvsdTi"] = xr.DataArray(derivative_array_v, coords=[layer_index, polarization_index])
        other_data["dTBhsdTi"] = xr.DataArray(derivative_array_h, coords=[layer_index, polarization_index])

        return make_result(sensor, np.transpose((Tbv, Tbh)), coords, other_data=other_data)
