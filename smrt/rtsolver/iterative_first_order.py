# coding: utf-8

"""
Provide the iterative first-order radiative transfer solver.

This module implements a first-order iterative solution for the radiative transfer equation
to calculate backscatter coefficients. The solver computes both zeroth and first-order
backscatter contributions and is computationally more efficient than the DORT solver,
but should be used with caution.

Key Features:
        - Separate different backscatter mechanisms for analysis.
        - More computationally efficient than full DORT solver.
        - Provide diagnostic information about scattering contributions.

Backscatter Components:
    Zeroth Order:
        Should be zero for flat interfaces and off-nadir angles. Represents the incident intensity that attenuates exponentially inside 
        the medium. Scattering is only included through its contribution to extinction.
        (Reference: Ulaby et al. 2014, first term of Eq. 11.74)

    First Order:
        Calculate three main contributions (Ulaby et al. 2014, Eqs. 11.75 and 11.62):
            - direct_backscatter: Single volume backscatter upwards by the layer.
            - double_bounce: Single volume backscatter downward by the layer and double specular reflection by the boundary.
            - reflected_scattering: 2× single bistatic scattering by the layer and single reflection by the lower boundary.

Usage:
    Basic usage with default settings and iba emmodel:
        >>> m = make_model("iba", "iterative_first_order")

    Return individual contributions for analysis:
        >>> m = make_model("iba", "iterative_first_order", rtsolver_options= {'return_contributions' : True})

    Handle errors gracefully in batch processing:
        >>> m = make_model("iba", "iterative_first_order", rtsolver_options = {'error_handling':'nan'}

Note:
    - This solver is designed for backscatter calculations only.
    - Single scattering albedo should be < 0.5 for reliable results. Can compare to dort to estimate unaccounted scattering.
    - Multiple scattering effects are not accounted for in first-order approximation.
    - For higher scattering albedos, a second-order solution is required.

References:
    Ulaby, F. T., et al. (2014). Microwave radar and radiometric remote sensing. Chapter 11.

    Tsang, L., Pan, J., Liang, D., Li, Z., Cline, D. W., & Tan, Y. (2007). Modeling Active Microwave Remote Sensing of Snow
    Using Dense Media Radiative Transfer (DMRT) Theory With Multiple-Scattering Effects. IEEE Transactions on Geoscience and
    Remote Sensing, 45(4), 990–1004. https://doi.org/10.1109/tgrs.2006.888854

    Tan, S., Chang, W., Tsang, L., Lemmetyinen, J., & Proksch, M. (2015). Modeling Both Active and Passive Microwave Remote
    Sensing of Snow Using Dense Media Radiative Transfer (DMRT) Theory With Multiple Scattering and Backscattering
    Enhancement. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 8(9), 4418–4430.
    https://doi.org/10.1109/jstars.2015.2469290
"""


# Stdlib import

# other import
import numpy as np
import xarray as xr

# local import
from smrt.core.error import SMRTError, smrt_warn
from smrt.core.lib import smrt_matrix, is_equal_zero
from smrt.core.result import make_result, prepare_kskaeps_profile_information
from smrt.core.fresnel import snell_angle


class IterativeFirstOrder(object):
    """
    Implement the iterative radiative transfer solver using first-order approximation.

    This solver computes the zeroth and first-order terms of the iterative solution
    for the radiative transfer equation. It provides efficient backscatter calculations
    but is limited to scenarios with relatively low scattering albedos.

    Args:
        error_handling: If set to "exception" (the default), raise an exception in case of error, stopping the code.
            If set to "nan", return a nan, so the calculation can continue, but the result is of course unusuable and
            the error message is not accessible. This is only recommended for long simulations that sometimes produce an error.

        return_contributions: If False (default), returns only total backscatter.
            If True, returns individual contributions:
                - 'direct_backscatter': Single volume backscatter upwards.
                - 'reflected_scattering': Bistatic scattering + single boundary reflection.
                - 'double_bounce': Volume backscatter + double boundary reflection.
                - 'zeroth': Zeroth-order contribution.
                - 'total': Sum of all contributions.
    """
    # Dimensions that this solver can handle directly:
    _broadcast_capability = {"theta_inc", "polarization_inc", "theta", "polarization"}

    def __init__(self, error_handling="exception", return_contributions=False):
        self.error_handling = error_handling
        self.return_contributions = return_contributions

    def solve(self, snowpack, emmodels, sensor, atmosphere=None):
        """
        Solve the radiative transfer equation for a given snowpack, emmodels and sensor configuration.

        Args:
            snowpack: Snowpack object.
            emmodels: List of electromagnetic models object.
            sensor: Sensor object.
            atmosphere: [Optional] Atmosphere object.

        Returns:
            result: Result object.
        """
        if sensor.mode != "A":
            raise SMRTError(
                "the iterative_rt solver is only suitable for activate microwave. " "Use an adequate sensor falling in this catergory."
            )

        if atmosphere is not None:
            raise SMRTError(
                "the iterative_rt solver can not handle atmosphere yet. " "Please put an issue on github if this feature is needed."
            )

        thickness = snowpack.layer_thicknesses
        temperature = snowpack.profile("temperature")

        effective_permittivity = [emmodel.effective_permittivity() for emmodel in emmodels]

        substrate = snowpack.substrate
        if substrate is not None and substrate.permittivity(sensor.frequency) is not None:
            substrate_permittivity = substrate.permittivity(sensor.frequency)
            if substrate_permittivity.imag < 1e-8:
                smrt_warn("the permittivity of the substrate has a too small imaginary part for reliable results")
                thickness.append(1e10)
                temperature.append(snowpack.substrate.temperature)
        else:
            substrate = snowpack.substrate
            substrate_permittivity = None

        # Active sensor
        # only V and H are necessary for first order
        self.pola = ["V", "H"]
        self.npol = len(self.pola)
        self.nlayer = snowpack.nlayer
        self.temperature = None

        # effective_permittivity = np.array(effective_permittivity)

        mu0 = np.cos(sensor.theta)
        self.dphi = np.pi

        # solve with first order iterative solution
        I = self.compute_intensity(
            snowpack,
            emmodels,
            sensor,
            snowpack.interfaces,
            substrate,
            effective_permittivity,
            mu0,
        )

        #  describe the results list of (dimension name, dimension array of value)
        coords = [("theta_inc", sensor.theta_inc_deg), ("polarization_inc", self.pola), ("polarization", self.pola)]

        # store other diagnostic information
        other_data = prepare_kskaeps_profile_information(snowpack, emmodels, effective_permittivity=effective_permittivity, mu=mu0)

        # get total intensity from the three contributions
        # first index is the number of mu
        total_I = I[0] + I[1] + I[2] + I[3]

        if self.return_contributions:
            # add total to the intensity array
            intensity = np.array([total_I, I[0], I[1], I[2], I[3]])
            return make_result(
                sensor,
                intensity,
                coords=[
                    (
                        "contribution",
                        ["total", "direct_backscatter", "reflected_scattering", "double_bounce", "zeroth"],
                    )
                ]
                + coords,
                other_data=other_data,
            )
        else:
            return make_result(sensor, total_I, coords=coords, other_data=other_data)

    def compute_intensity(self, snowpack, emmodels, sensor, interfaces, substrate, effective_permittivity, mu0):
        # """
        # Calculate the intensity contributions for first-order backscatter.

        # Computes zeroth and first-order backscatter contributions including direct
        # backscatter, reflection backscatter, and double bounce scattering.

        # Args:
        #     snowpack (Snowpack): Snowpack object with layer properties.
        #     emmodels (list): List of electromagnetic models for each layer.
        #     sensor (Sensor): Sensor configuration.
        #     interfaces (list): List of interface objects.
        #     substrate (Substrate): Substrate object.
        #     effective_permittivity (list): Complex permittivity for each layer.
        #     mu0 (array_like): Cosine of incident angles.

        # Returns:
        #     np.ndarray: Intensity array with shape (4, n, npol, npol) containing:
        #         - [0]: Direct backscatter contribution
        #         - [1]: Reflected scattering contribution
        #         - [2]: Double bounce contribution
        #         - [3]: Zeroth order contribution
        # """
        nlayer = snowpack.nlayer
        dphi = self.dphi
        n = len(mu0)  # number of streams
        npol = self.npol

        # no need for 3x3 in first order solution
        I_i = np.array([[1, 0], [0, 1]])

        interface_l = _InterfaceProperties(
            sensor.frequency,
            interfaces,
            substrate,
            effective_permittivity,
            mu0,
            npol,
            nlayer,
            dphi,
        )

        # get list of thickness for each layer
        thickness = snowpack.layer_thicknesses

        # mu for all layer and can have more than 1 if theta from sensor is a list
        mus = interface_l.mu

        # refraction_factor for first layer eq 22a and eq 22b in Tsang et al 2007
        refraction_factor_0 = ((1 / effective_permittivity[0].real) * (mu0 / mus[0]))[:, np.newaxis, np.newaxis]

        # Intensity incident transmitted to first layer from air
        I_l = _get_np_matrix(interface_l.Tbottom_coh[-1], npol, n) @ I_i * refraction_factor_0

        # 3 for the number of contribution for the first order backscatter
        intensity_up = np.zeros((4, n, npol, npol))
        optical_depth = 0
        for l in range(nlayer):
            # check scat albedo for validity of iterative solution
            ke = emmodels[l]._ks + emmodels[l].ka
            scat_albedo = emmodels[l]._ks / (ke)
            if scat_albedo > 0.5:
                smrt_warn(
                    f"Warning : scattering albedo ({np.round(scat_albedo, 2)}) might be too high"
                    + " for iterative method. Limit is around 0.5."
                )

            # prepare matrix of interface
            # transmission matrix of the top layer to l-1
            Ttop_coh_m = _get_np_matrix(interface_l.Ttop_coh[l], npol, n)

            # transmission matrix of the bottom layer to l+1
            Tbottom_coh_m = _get_np_matrix(interface_l.Tbottom_coh[l], npol, n)

            # Specular Reflection matrix of the bottom layer
            Rbottom_coh_m = _get_np_matrix(interface_l.Rbottom_coh[l], npol, n)

            # Diffuse reflection matrix of the bottom layer
            Rbottom_diff_m = _get_np_matrix(interface_l.Rbottom_diff[l], npol, n)

            # get phase function for array of mu and -mu
            mus_sym = np.concatenate([-mus[l], mus[l]])
            # Note that the phase calculation is inefficiant as it calculates all combinations of (mus_sym, mus_sym).
            # Solving this issue would require to rewrite all the phase methods. For the future, maybe.
            phases = emmodels[l].phase(mus_sym, mus_sym, dphi, npol).values

            # 1/4pi normalization of the RT equation for SMRT
            # applied to phase here, interface and substrate already have the smrt_norm
            phases /= 4 * np.pi

            def reshape_phase(p):
                return np.rollaxis(p.diagonal(axis1=-2, axis2=-1), -1)  # take the diagonal of the last two axes

            P_Up = reshape_phase(phases[:, :, 0, 0:n, n:])  # P(-mu, mu)
            P_Down = reshape_phase(phases[:, :, 0, n:, 0:n])  # P(mu, -mu)
            P_Bi_Up = reshape_phase(phases[:, :, 0, n:, n:])  # P(mu, mu)
            P_Bi_Down = reshape_phase(phases[:, :, 0, 0:n, 0:n])  # P(-mu, -mu)

            layer_optical_depth = ke * thickness[l]
            optical_depth += layer_optical_depth

            # convert to 3d array for computation of intensity
            # allow computation of incident angle
            # two way attenuation (ulaby et al 2014, eq: 11.2)
            mus_l = mus[l][:, np.newaxis, np.newaxis]
            gammas2 = np.exp(-2 * layer_optical_depth / mus_l)

            """
            Zeroth order, ulaby et al 2014 (first term of 11.74)
            Simply the reduced incident intensity, which attenuates exponentially inside the medium.
            Scattering is not included, except for its contribution to extinction.
            Should be zero for flat interface and off-nadir.
            """
            I0_mu = Ttop_coh_m**2 @ (gammas2 * (Rbottom_diff_m @ I_l))

            """
            First order, ulaby et al 2014 (11.75 and 11.62 )
            Four contributions are taken into account
            - Single volume backscatter upwards by the layer (direct backscatter)
            - 2x single bistatic scattering by the layer and single reflection by the lower boundary. (reflected scattering)
            - Single volume backscatter downward by the layer and double specular reflection by the boundary (double bounce)

            """

            I1_back = Ttop_coh_m**2 @ ((1 - gammas2) / (2 * ke) * P_Up) @ I_l

            I1_2B = Ttop_coh_m**2 @ (((1 - gammas2) / (2 * ke) * gammas2) * (Rbottom_coh_m @ P_Down @ Rbottom_coh_m)) @ I_l

            I1_ref_scat = Ttop_coh_m**2 @ (thickness[l] * gammas2 / mus_l * (P_Bi_Down @ Rbottom_coh_m + Rbottom_coh_m @ P_Bi_Up)) @ I_l

            # shape of intensity (incident angle, first order contribution, npo, npol)
            I1 = np.array([I1_back, I1_ref_scat, I1_2B, I0_mu]).reshape(4, n, npol, npol)

            # add intensity
            intensity_up += I1

            if l < nlayer - 1:
                mus_l1 = mus[l + 1][:, np.newaxis, np.newaxis]
                # refraction factor for layer l, eq 22a and eq 22b in Tsang et al 2007
                refraction_factor_l = (effective_permittivity[l].real / effective_permittivity[l + 1].real) * (mus_l / mus_l1)
                # intensity in the layer transmitted downward for upper layer with one way attenuation
                I_l = Tbottom_coh_m @ (gammas2 * refraction_factor_l * I_l)

        if substrate is None and optical_depth < 5:
            smrt_warn(
                "The solver has detected that the snowpack is optically shallow (tau=%g) and no substrate has"
                "been set, meaning that the space under the snowpack is vaccum and that the snowpack is"
                "shallow enough to affect the signal measured at the surface. This is usually not wanted. "
                "Either increase the thickness of the snowpack or set a substrate."
                "If wanted, add a transparent substrate to supress this warning" % optical_depth
            )

        # snow to air final transmission upward
        intensity = _get_np_matrix(interface_l.Ttop_coh[0], npol, n) @ intensity_up

        # 1/4pi normalization of the RT equation like DORT
        return intensity


def _get_np_matrix(smrt_m, npol, n_mu):
    # """
    # Convert SMRT matrix format to numpy array format.

    # Args:
    #     smrt_m (SMRTMatrix): SMRT matrix object to convert.
    #     npol (int): Number of polarizations.
    #     n_mu (int): Number of incident angles.

    # Returns:
    #     np.ndarray: Numpy array of shape (n_mu, npol, npol).

    # Raises:
    #     SMRTError: If matrix type is not supported for conversion.
    # """
    if is_equal_zero(smrt_m):
        # zero matrix
        np_m = np.zeros((n_mu, npol, npol))
        return np_m

    elif smrt_m.mtype.startswith("diagonal"):
        # diagonal matrix
        np_m = np.zeros((n_mu, npol, npol))
        for i in range(n_mu):
            np.fill_diagonal(np_m[i], smrt_m.diagonal[:, i])
        return np_m

    elif smrt_m.mtype.startswith("dense"):
        # dense matrix
        np_m = np.array([smrt_m.values[:, :, 0, i, i] for i in range(n_mu)])
        return np_m

    else:
        SMRTError("SMRT matrix conversion type to numpy conversion not implemented in iterative solver")


class _InterfaceProperties(object):
    # """
    # Helper class to organize interface properties for multi-layer snowpack.

    # Manages transmission and reflection matrices for coherent and diffuse
    # scattering at interfaces between layers and boundaries.
    # Prepare interface properties of multi-layer snowpack layer l
    # Index -1 refers to air layer

    # Args:
    #     frequency (float): Electromagnetic frequency.
    #     interfaces (list): List of interface objects.
    #     substrate (Substrate): Substrate object.
    #     permittivity (list): Complex permittivity for each layer.
    #     mu0 (array_like): Cosine of incident angles.
    #     npol (int): Number of polarizations.
    #     nlayer (int): Number of layers.
    #     dphi (float): Azimuth angle difference.

    # Attributes:
    #     Rtop_coh (dict): Coherent reflection matrices for top interfaces.
    #     Rtop_diff (dict): Diffuse reflection matrices for top interfaces.
    #     Ttop_coh (dict): Coherent transmission matrices for top interfaces.
    #     Rbottom_coh (dict): Coherent reflection matrices for bottom interfaces.
    #     Rbottom_diff (dict): Diffuse reflection matrices for bottom interfaces.
    #     Tbottom_coh (dict): Coherent transmission matrices for bottom interfaces.
    #     mu (dict): Cosine of refraction angles for each layer.
    # """

    def __init__(self, frequency, interfaces, substrate, permittivity, mu0, npol, nlayer, dphi):

        self.Rtop_coh = dict()
        self.Rtop_diff = dict()
        self.Ttop_coh = dict()
        # self.Ttop_diff = dict()
        self.Rbottom_coh = dict()
        self.Rbottom_diff = dict()
        self.Tbottom_coh = dict()
        # self.Tbottom_diff = dict()

        self.mu = dict()
        # air-snow DOWN
        # index -1 refers to air layer
        self.Tbottom_coh[-1] = interfaces[0].coherent_transmission_matrix(frequency, 1, permittivity[0], mu0, npol)

        # air-snow DOWN
        self.Rbottom_coh[-1] = interfaces[0].specular_reflection_matrix(frequency, 1, permittivity[0], mu0, npol)
        self.Rbottom_diff[-1] = (
            interfaces[0].diffuse_reflection_matrix(frequency, 1, permittivity[0], mu0, mu0, dphi, npol)
            if hasattr(interfaces[0], "diffuse_reflection_matrix")
            else smrt_matrix(0)
        )

        for l in range(nlayer):
            # define permittivity
            # #for permittivity, index 0 = air, length of permittivity is l+1
            eps_lm1 = permittivity[l - 1] if l > 0 else 1
            eps_l = permittivity[l]
            if l < nlayer - 1:
                eps_lp1 = permittivity[l + 1]
            else:
                eps_lp1 = None

            self.mu[l] = snell_angle(1, eps_l, mu0)

            self.Rtop_coh[l] = interfaces[l].specular_reflection_matrix(frequency, eps_l, eps_lm1, self.mu[l], npol)

            self.Rtop_diff[l] = (
                interfaces[l].diffuse_reflection_matrix(frequency, eps_l, eps_lm1, self.mu[l], self.mu[l], dphi, npol)
                if hasattr(interfaces[l], "diffuse_reflection_matrix")
                else smrt_matrix(0)
            )

            self.Ttop_coh[l] = interfaces[l].coherent_transmission_matrix(frequency, eps_l, eps_lm1, self.mu[l], npol)

            if l < nlayer - 1:
                # set up interfaces
                # snow - snow
                # Upward
                self.Rbottom_coh[l] = interfaces[l + 1].specular_reflection_matrix(frequency, eps_l, eps_lp1, self.mu[l], npol)

                # other than flat interface
                self.Rbottom_diff[l] = interfaces[l + 1].diffuse_reflection_matrix(
                    frequency, eps_l, eps_lp1, self.mu[l], self.mu[l], dphi, npol
                )

                self.Tbottom_coh[l] = interfaces[l + 1].coherent_transmission_matrix(frequency, eps_l, eps_lp1, self.mu[l], npol)

            elif substrate is not None:
                self.Rbottom_coh[l] = substrate.specular_reflection_matrix(frequency, eps_l, self.mu[l], npol)

                self.Rbottom_diff[l] = (
                    substrate.diffuse_reflection_matrix(frequency, eps_l, self.mu[l], self.mu[l], dphi, npol)
                    if hasattr(substrate, "diffuse_reflection_matrix")
                    else smrt_matrix(0)
                )

                # sub-snow
                self.Tbottom_coh[l] = smrt_matrix(0)

            else:
                # fully transparent substrate
                self.Rbottom_coh[l] = smrt_matrix(0)
                self.Rbottom_diff[l] = smrt_matrix(0)
                self.Tbottom_coh[l] = smrt_matrix(0)
