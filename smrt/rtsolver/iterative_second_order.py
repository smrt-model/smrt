# coding: utf-8

"""
Provide the iterative second-order radiative transfer solver.

This module implements a second-order iterative solution for the radiative transfer equation
to calculate backscatter coefficients. The solver computes both zeroth, first-order, second
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

    Second Order:
        Calculate three main contributions of second order (Karam et al. 1995):
            - double_scattering_intralayer: Double scattering of the snow microstructure inside one layer.
            - ground_layer_scattering: Substrate diffuse reflection with one snow microstructure scattering.
            - double_scattering_interlayer: Double scattering of the snow microstructure between all two layer combinations. 

Usage:
    Basic usage with default settings and iba emmodel:
        >>> m = make_model("iba", "iterative_second_order")

    Return individual contributions for analysis:
        >>> m = make_model("iba", "iterative_second_order", rtsolver_options= {'return_contributions' : True})


Note:
    - This solver is designed for backscatter calculations only.
    - Single scattering albedo should be < 0.5 for reliable results. Can compare to dort to estimate unaccounted scattering.
    - Using the interlayer double scattering with multiple layers (> 10) is not suggested (increased computation)

References:
    Karam, M. A, Amar, F., Fung, A., Mougin, E. Lopes, A., Le Vine, D. M., & Beaudoin, A. (1995). A microwave polarimetric 
    scattering model for forest canopies based on vector radiative transfer theory. Remote Sensing of Environment, 53(1), 16-30.
    https://doi.org/10.1016/0034-4257(95)00048-6

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
import numpy as np
import xarray as xr
#from numba import njit

# local import
from smrt.core.error import SMRTError, smrt_warn
from smrt.core.result import make_result
from smrt.rtsolver.rtsolver_utils import RTSolverBase, prepare_kskaeps_profile_information
from smrt.rtsolver.streams import compute_stream
from smrt.rtsolver.iterative_first_order import IterativeFirstOrder, _InterfaceProperties, compute_gamma, compute_refraction_factor


class IterativeSecondOrder(RTSolverBase):
    """
    Implement the iterative radiative transfer solver using second-order approximation.

    This solver computes the zeroth first-order and second-order terms of the iterative solution
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
                - 'double_scattering_intralayer': Double scattering of the snow microstructure inside one layer.
                - 'ground_layer_scattering' : Substrate diffuse reflection with one snow microstructure scattering.
                - 'double_scattering_interlayer' : Double scattering of the snow microstructure between two layer. Calculate all 
                combinations possible.
                - 'total': Sum of all contributions.
        n_max_stream: Stream for integral of theta 0 to pi/2.
        stream_mode: If set to "most_refringent" (the default) or "air", streams arecalculated using the Gauss-Legendre 
            polynomials and then use Snell-law to prograpate the direction in the other layers. If set to "uniform_air", 
            streams are calculated uniformly in air and then according to Snells law.
        m_max: Number of mode for integral of phi from 0 to 2pi.
        interlayer: If False (default), skip interlayer calculation. Using interlayer with a large number of layer will increase 
            computation time significantly.
        compute_substrate_integral: If True (default), compute 'ground_layer_scattering'.
    """

    # this specifies which dimension this solver is able to deal with.
    #  Those not in this list must be managed by the called (Model object)
    # e.g. here, frequency, time, ... are not managed
    _broadcast_capability = {"theta_inc", "polarization_inc", "theta", "polarization"}

    def __init__(
        self,
        error_handling="exception",
        return_contributions=False,
        n_max_stream=32,  # stream for integral of theta 0 to pi/2
        stream_mode="most_refringent",
        m_max=5,  # mode for integral of phi from 0 to 2pi
        interlayer = False, #using interlayer with a large number of layer will increase computation time significantly
        compute_substrate_integral = True
    ):
        self.error_handling = error_handling
        self.return_contributions = return_contributions
        self.n_max_stream = n_max_stream
        self.stream_mode = stream_mode
        self.m_max = m_max
        self.interlayer = interlayer
        self.compute_substrate_integral = compute_substrate_integral

    def solve(self, snowpack, emmodels, sensor, atmosphere=None):
        # """solve the radiative transfer equation for a given snowpack, emmodels and sensor configuration."""
        if sensor.mode != "A":
            raise SMRTError(
                "the iterative_rt solver is only suitable for activate microwave. Use an adequate sensor falling in" + "this catergory."
            )

        if atmosphere is not None:
            raise SMRTError(
                "the iterative_rt solver can not handle atmosphere yet. Please put an issue on github if this" + "feature is needed."
            )

        self.init_solve(snowpack, emmodels, sensor, atmosphere)
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

        thickness = snowpack.layer_thicknesses
        temperature = snowpack.profile("temperature")

        # Active sensor
        # only returns V and H but U is used in the calculation. U is removed at the end to match first order
        self.pola = ["V", "H"]
        # but U is used in the calculation, so npol =3
        self.npol = 3
        self.nlayer = snowpack.nlayer
        temperature = None
        mu0 = np.cos(sensor.theta)
        self.len_mu = len(mu0)
        self.dphi = np.pi

        # # create the model
        # self.emmodels = emmodels
        # # interface
        # self.interfaces = snowpack.interfaces
        # # sensor
        # self.sensor = sensor

        # get stream for integral of theta for 0 to pi/2
        streams = compute_stream(self.n_max_stream, self.effective_permittivity, mode=self.stream_mode)

        # get the first order
        first_solver = IterativeFirstOrder(return_contributions=True)
        # 2 pol for first order
        first_solver.npol = 2
        first_solver.dphi = self.dphi
        I1 = first_solver.compute_intensity(snowpack, emmodels, sensor, snowpack.interfaces, substrate, self.effective_permittivity, mu0)
        total_I1 = I1[0] + I1[1] + I1[2] + I1[3]

        # solve the second order iterative solution
        I2_1, I2_2, I2_3 = self.compute_intensity(snowpack, emmodels, streams, sensor, self.effective_permittivity, mu0)

        # add first and second, get rid of U pol for second order so it match first order
        total_I = total_I1 + I2_1[:, 0:2, 0:2] + I2_2[:, 0:2, 0:2] + I2_3[:, 0:2, 0:2]
        #  describe the results list of (dimension name, dimension array of value)
        coords = [("theta_inc", sensor.theta_inc_deg), ("polarization_inc", self.pola), ("polarization", self.pola)]

        # store other diagnostic information
        other_data = prepare_kskaeps_profile_information(snowpack, emmodels, effective_permittivity=self.effective_permittivity, mu=mu0)

        if self.return_contributions:
            # add total to the intensity array
            intensity = np.array([total_I, I1[0], I1[1], I1[2], I1[3], I2_1[:, 0:2, 0:2], I2_2[:, 0:2, 0:2], I2_3[:, 0:2, 0:2]])
            return make_result(
                sensor,
                intensity,
                coords=[
                    (
                        "contribution", ["total", 
                                         "direct_backscatter", 
                                         "reflected_scattering", 
                                         "double_bounce",
                                         "zeroth", 
                                         "double_scattering_intralayer",
                                         "ground_layer_scattering",
                                         "double_scattering_interlayer"],
                    )
                ]
                + coords,
                other_data=other_data,
            )
        else:
            return make_result(sensor, total_I, coords=coords, other_data=other_data)

    def compute_intensity(self, snowpack, emmodels, streams, sensor, effective_permittivity, mu0):
        # mode for integral of phi from 0 to 2pi
        npol = self.npol
        dphi = self.dphi
        len_mu = self.len_mu
        nlayer = snowpack.nlayer
        thickness = snowpack.layer_thicknesses
        substrate = snowpack.substrate

        interface_l = _InterfaceProperties(
            sensor.frequency, snowpack.interfaces, substrate, effective_permittivity, mu0, npol, nlayer, dphi
        )

        # check if layer to ground interaction can to be calculated if True, need bistatic coef from diffuse matrix
        if self.compute_substrate_integral:
            if (substrate is not None) and (hasattr(substrate, "ft_even_diffuse_reflection_matrix")):
                try:   
                    #try for a different angle
                    substrate.ft_even_diffuse_reflection_matrix(sensor.frequency, 
                                                                complex(16,0.1), 
                                                                mu0, 
                                                                mu0 + 0.5, 
                                                                self.m_max, 
                                                                self.npol)
                    self.compute_substrate_integral = True
                except:
                    self.compute_substrate_integral = False
            else:
                self.compute_substrate_integral = False

        # mu for all layer and can have more than 1 if theta from sensor is a list
        mus = interface_l.mu

        I_i = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]]).T


        # intensity in the layer
        # dense snow factor (I think) eq 22a and eq 22b in Tsang et al 2007
        refraction_factor_0 = compute_refraction_factor(1, effective_permittivity[0], mu0, mus[0])[:, np.newaxis, np.newaxis]
        transmission_bottom_surface = interface_l.transmission_bottom(-1)
        I_l = transmission_bottom_surface @ I_i * refraction_factor_0

        # needs for loop for multiple layers
        optical_depth = 0
        intensity_up_intra = np.zeros((len(mu0), npol, npol))
        intensity_up_ground = np.zeros((len(mu0), npol, npol))
        intensity_up_inter = np.zeros((len(mu0), npol, npol))
        for ln in range(nlayer):
            # prepare matrix of interface
            # transmission matrix of the top layer to l-1 in to l
            transmission_top = interface_l.transmission_top(ln)

            # transmission matrix of the layer l to l+1
            transmission_bottom = interface_l.transmission_bottom(ln)

            # stream for integral
            mu_int_ln = streams.mu[ln][::-1]
            weight_ln = streams.weight[ln][::-1]

            # extinction coef and layer optical depth
            ke_ln = emmodels[ln]._ks + emmodels[ln].ka
            layer_optical_depth_ln = ke_ln * thickness[ln]
            optical_depth += layer_optical_depth_ln
            

            # get intensity of double scatter 
            intensity_up_intra += transmission_top @ self.compute_double_scattering_intralayer(emmodels[ln], 
                                                                                            I_l, 
                                                                                            weight_ln, 
                                                                                            mu_int_ln, 
                                                                                            mus[ln], 
                                                                                            ke_ln, 
                                                                                            layer_optical_depth_ln)

            if self.compute_substrate_integral:
                #reflection matrix of the ground
                Rbottom_diff_int = get_substrate_reflectivity_integral(substrate, 
                                                                        sensor.frequency, 
                                                                        effective_permittivity[ln], 
                                                                        np.concatenate([-mus[ln], mus[ln]]), 
                                                                        np.concatenate([-mu_int_ln, mu_int_ln]),
                                                                        self.m_max,
                                                                        npol)
                    
                #optical dept of layer n+1 to the ground
                layer_optical_depth_ln_ground = np.sum([(emmodels[lng]._ks + emmodels[lng].ka) * thickness[lng] 
                                                        for lng in range(ln+1, nlayer)])
                
                # #calculate the cumulative refraction factor in between layer n and ground
                # cumulative_refraction_factor = np.prod([compute_refraction_factor(effective_permittivity[lng], 
                #                                                           effective_permittivity[lng + 1], 
                #                                                           mus[lng], 
                #                                                           mus[lng+1]) for lng in range(ln+1, nlayer-1)])

                intensity_up_ground += transmission_top @ self.compute_scattering_layer_ground(emmodels[ln], 
                                                                                    I_l, 
                                                                                    weight_ln, 
                                                                                    mu_int_ln, 
                                                                                    mus[ln], 
                                                                                    ke_ln, 
                                                                                    layer_optical_depth_ln,
                                                                                    layer_optical_depth_ln_ground, 
                                                                                    Rbottom_diff_int)

            if self.interlayer:
                # r is a layer located betwenn n and m, to compute cumulative attenuation between layer n and m
                layer_optical_depth_lr = layer_optical_depth_ln
                #scattering interlayer
                for lm in range(1, nlayer):
                    if ln == lm or ln > lm:
                        continue
                    else:
                        # stream for integral
                        mu_int_lm = streams.mu[lm][::-1]
                        weight_lm = streams.weight[lm][::-1]

                        # extinction coef and layer optical depth
                        ke_lm = emmodels[lm]._ks + emmodels[lm].ka
                        layer_optical_depth_lm = ke_lm * thickness[lm]
                        layer_optical_depth_lr += layer_optical_depth_lm

                        intensity_up_inter += transmission_top @ self.compute_double_scattering_interlayer(emmodels[ln],
                                                                                                emmodels[lm], 
                                                                                                I_l,
                                                                                                weight_ln, 
                                                                                                mu_int_ln, 
                                                                                                mus[ln], 
                                                                                                weight_lm, 
                                                                                                mu_int_lm, 
                                                                                                mus[lm], 
                                                                                                ke_ln, 
                                                                                                ke_lm, 
                                                                                                layer_optical_depth_ln, 
                                                                                                layer_optical_depth_lm,
                                                                                                layer_optical_depth_lr)


            # intensity transmitted down to next layer
            gamma2_ln = compute_gamma(mus[ln], layer_optical_depth_ln) ** 2
            #transmitted intensity
            if ln < nlayer - 1:
                # dense snow factor (I think) eq 22a and eq 22b in Tsang et al 2007
                # dense_factor_ln = ((effective_permittivity[ln].real / effective_permittivity[ln + 1].real) \
                #     * (mus[ln] / interface_l.mu[ln + 1]))[:, np.newaxis, np.newaxis]
                refraction_factor_ln = compute_refraction_factor(effective_permittivity[ln], 
                                                                effective_permittivity[ln + 1],
                                                                mus[ln],
                                                                mus[ln + 1])[:, np.newaxis, np.newaxis]
                
                # intensity in the layer transmitted downward for upper layer with one way attenuatio
                I_l = transmission_bottom @ (I_l * gamma2_ln) * refraction_factor_ln



                if snowpack.substrate is None and optical_depth < 5:
                    smrt_warn(
                        "The solver has detected that the snowpack is optically shallow (tau=%g) and no substrate has been set, meaning that the space "
                        "under the snowpack is vaccum and that the snowpack is shallow enough to affect the signal measured at the surface."
                        "This is usually not wanted. Either increase the thickness of the snowpack or set a substrate."
                        " If wanted, add a transparent substrate to supress this warning" % optical_depth
                    )

        return intensity_up_intra, intensity_up_ground, intensity_up_inter


    def compute_double_scattering_intralayer(self, emmodel, I_l, weight, mu_int, mus_i, ke, layer_optical_depth):
        #eqn A11 Karam et al 1995
        # A and B are functions of theta, define explicitely in the book
        # integral of phi and theta
        # Compute summation of the Gauss quadrature (integral of theta) G.Picard thesis eq. 2.22
        # integral of theta from 0 to 1, for mu with change of variable
        # need weight and mu_int from streams

        # multiple incident angle
        m_max = self.m_max
        len_mu = self.len_mu
        npol = self.npol
        dphi = self.dphi

        # get negative and positive mu
        mu_i_sym = np.concatenate([-mus_i, mus_i])
        mus_int = np.concatenate([-mu_int, mu_int])

        n_stream = len(mu_int)
        n_mu_i = len(mus_i)

        phase_mu_int_mu = emmodel.ft_even_phase(mus_int, mu_i_sym, self.m_max) / (4 * np.pi)
        phase_mu_mu_int = emmodel.ft_even_phase(mu_i_sym, mus_int, self.m_max) / (4 * np.pi)

        P1 = phase_mu_mu_int[:, :, :, n_mu_i:, n_stream:]  # P(mu_i, mu_int)
        P2 = phase_mu_int_mu[:, :, :, n_stream:, 0:n_mu_i]  # P(mu_int, -mu_i)
        P3 = phase_mu_mu_int[:, :, :, n_mu_i:, 0:n_stream]  # P(mu_i, -mu_int)
        P4 = phase_mu_int_mu[:, :, :, n_stream:, n_mu_i:]  # P(mu_int, mu_i)

        P5 = phase_mu_mu_int[:, :, :, n_mu_i:, 0:n_stream]  # P(mu_i, -mu_int)
        P6 = phase_mu_int_mu[:, :, :, 0:n_stream, 0:n_mu_i]  # P(-mu_int, -mu_i)
        P7 = phase_mu_mu_int[:, :, :, n_mu_i:, n_stream:]  # P(mu_i, mu_int)
        P8 = phase_mu_int_mu[:, :, :, n_stream:, n_mu_i:]  # P(mu_int, mu_i)

        sum_a, sum_b = 0, 0
        for mu, w, i in zip(mu_int, weight, range(n_stream)):
            # bound 1 of the integral
            # integral of mu (G.Picard thesis p.72)
            # -1 coef for incident angle
            # P(mu_i, mu_int)* P(mu_int, -mu_i) + P(mu_i, -mu_int)* P(mu_int, mu_i)

            A = compute_A(mus_i, mu, ke, layer_optical_depth)
            sum_a += w * (
                    (A * compute_int_phi(P1[:, :, :, :, i], P2[:, :, :, i, :], m_max, len_mu, npol, dphi))
                    + (A * compute_int_phi(P3[:, :, :, :, i], P4[:, :, :, i, :], m_max, len_mu, npol, dphi))
                )

            # P(mu_i, -mu)* P(-mu_int, -mu_i) + P(mu_i, mu_int)* P(mu_int, mu_i)
            B = compute_B(mus_i, mu, ke, layer_optical_depth)
            sum_b += w * (
                    (B * compute_int_phi(P5[:, :, :, :, i], P6[:, :, :, i, :], m_max, len_mu, npol, dphi))
                    + (B * compute_int_phi(P7[:, :, :, :, i], P8[:, :, :, i, :], m_max, len_mu, npol, dphi))
                )

        I_mu = (sum_a + sum_b) @ I_l

        return I_mu

    def compute_scattering_layer_ground(self, emmodel, I_l, weight, mu_int, mus_i, ke, layer_optical_depth, layer_optical_depth_ln_ground, Rbottom_diff_int):
        # volume-ground interaction contribution
        #eqn A8 Karam et al 1995

        # multiple incident angle
        m_max = self.m_max
        len_mu = self.len_mu
        npol = self.npol
        dphi = self.dphi

        # get negative and positive mu
        mu_i_sym = np.concatenate([-mus_i, mus_i])
        mu_int_sym = np.concatenate([-mu_int, mu_int])

        n_stream = len(mu_int)
        n_mu_i = len(mus_i)
        
        phase_mu_int_mu = emmodel.ft_even_phase(mu_int_sym, mu_i_sym, m_max) / (4 * np.pi)
        #phase_mu_mu_int = emmodel.ft_even_phase(mu_i_sym, mu_int_sym, m_max) / (4 * np.pi)

        R1 = Rbottom_diff_int['i_int'][:, :, :, n_mu_i:, n_stream:]  # R(mu_i, mu_int)
        P1 = phase_mu_int_mu[:, :, :, 0:n_stream, 0:n_mu_i]  # P(-mu_int, -mu_i)
        R2 = Rbottom_diff_int['i_int'][:, :, :, n_mu_i:, 0:n_stream]  # R(mu_i, -mu_int)
        P2 = phase_mu_int_mu[:, :, :, n_stream:, 0:n_mu_i]  # P(mu_int, -mu_i)

        sum_e, sum_f = 0, 0
        for mu, w, i in zip(mu_int, weight, range(n_stream)):
            # bound 1 of the integral
            # integral of mu (G.Picard thesis p.72)
            # -1 coef for incident angle
            # Rbottom(mu_i, mu_int) * P(-mu_int, -mu_i) + Rbottom(mu_i, -mu_int) * P(mu_int,-mu_i)

            E = compute_E(mus_i, mu, ke, layer_optical_depth, layer_optical_depth_ln_ground)
            sum_e += w * (
                    (E * compute_int_phi(R1[:, :, :, :, i], P1[:, :, :, i, :], m_max, len_mu, npol, dphi))
                    + (E * compute_int_phi(R2[:, :, :, :, i], P2[:, :, :, i, :], m_max, len_mu, npol, dphi))
                )

        return sum_e @ I_l
    

    def compute_double_scattering_interlayer(self,
                                            emmodel_ln,
                                            emmodel_lm, 
                                            I_l,
                                            weight_ln, 
                                            mu_int_ln, 
                                            mu_i_ln, 
                                            weight_lm, 
                                            mu_int_lm, 
                                            mu_i_lm, 
                                            ke_ln, 
                                            ke_lm, 
                                            layer_optical_depth_ln, 
                                            layer_optical_depth_lm,
                                            layer_optical_depth_lr):
        #scattering from the m layer with scattering from the n layer,
        # eqn A13a Karam et al 1995
        # integral of phi and theta
        # Compute summation of the Gauss quadrature (integral of theta) G.Picard thesis eq. 2.22
        # integral of theta from 0 to 1, for mu with change of variable
        # need weight and mu_int from streams

        # multiple incident angle

        m_max = self.m_max
        len_mu = self.len_mu
        npol = self.npol
        dphi = self.dphi

        # get negative and positive mu
        mu_i_phase_ln = np.concatenate([-mu_i_ln, mu_i_ln])
        mu_int_phase_ln = np.concatenate([-mu_int_ln, mu_int_ln])

        mu_i_phase_lm = np.concatenate([-mu_i_lm, mu_i_lm])
        mu_int_phase_lm = np.concatenate([-mu_int_lm, mu_int_lm])

        n_stream_ln = len(mu_int_ln)
        n_mu_i_ln = len(mu_i_ln)

        n_stream_lm = len(mu_int_lm)
        n_mu_i_lm = len(mu_i_lm)

        phase_mu_int_mu_ln = emmodel_ln.ft_even_phase(mu_int_phase_ln, mu_i_phase_ln, self.m_max) / (4 * np.pi)
        phase_mu_mu_int_ln = emmodel_ln.ft_even_phase(mu_i_phase_ln, mu_int_phase_ln, self.m_max) / (4 * np.pi)

        phase_mu_int_mu_lm = emmodel_lm.ft_even_phase(mu_int_phase_lm, mu_i_phase_lm, self.m_max) / (4 * np.pi)
        phase_mu_mu_int_lm = emmodel_lm.ft_even_phase(mu_i_phase_lm, mu_int_phase_lm, self.m_max) / (4 * np.pi)

        P1n = phase_mu_mu_int_ln[:, :, :, n_mu_i_ln:, n_stream_ln:]  # P(mu_i_ln, mu_int_ln)
        P2m = phase_mu_int_mu_lm[:, :, :, n_stream_lm:, 0:n_mu_i_lm]  # P(mu_int_ln, -mu_i_ln)
        P3n = phase_mu_mu_int_ln[:, :, :, n_mu_i_ln:, 0:n_stream_ln]  # P(mu_i_lm, -mu_int_lm)
        P4m = phase_mu_int_mu_lm[:, :, :, n_stream_lm:, n_mu_i_lm:]  # P(mu_int_lm, mu_i_lm)

        P5m = phase_mu_mu_int_lm[:, :, :, n_mu_i_lm:, 0:n_stream_lm]  # P(mu_i_lm, -mu_int_lm)
        P6n = phase_mu_int_mu_ln[:, :, :, 0:n_stream_ln, 0:n_mu_i_ln]  # P(-mu_int_lm, -mu_i_lm)
        P7m = phase_mu_mu_int_lm[:, :, :, n_mu_i_lm:, n_stream_lm:]  # P(mu_i_ln, mu_int_ln)
        P8n = phase_mu_int_mu_ln[:, :, :, n_stream_ln:, n_mu_i_ln:]  # P(mu_int_ln, mu_i_ln)
        
        sum_c, sum_d = 0, 0
        for mu_ln, w_ln, i_ln, mu_lm, w_lm, i_lm in zip(mu_int_ln, weight_ln, range(n_stream_ln), mu_int_lm, weight_lm, range(n_stream_lm)):
            # bound 1 of the integral
            # integral of mu (G.Picard thesis p.72)
            # -1 coef for incident angle
            # P(mu_i, mu_int)* P(mu_int, -mu_i) + P(mu_i, -mu_int)* P(mu_int, mu_i)

            C = compute_C(mu_i_ln, mu_ln, ke_ln, ke_lm, layer_optical_depth_ln, layer_optical_depth_lm, layer_optical_depth_lr)
            sum_c += w_ln * ((C * compute_int_phi(P1n[:, :, :, :, i_ln], P2m[:, :, :, i_lm, :], m_max, len_mu, npol, dphi)) + 
                        (C * compute_int_phi(P3n[:, :, :, :, i_ln], P4m[:, :, :, i_lm, :], m_max, len_mu, npol, dphi))
                    )

            # P(mu_i, -mu)* P(-mu_int, -mu_i) + P(mu_i, mu_int)* P(mu_int, mu_i)
            D = compute_D(mu_i_ln, mu_ln, ke_ln, ke_lm, layer_optical_depth_ln, layer_optical_depth_lm, layer_optical_depth_lr)
            sum_d += w_ln * ((D * compute_int_phi(P5m[:, :, :, :, i_lm], P6n[:, :, :, i_ln, :], m_max, len_mu, npol, dphi)) + 
                        (D * compute_int_phi(P7m[:, :, :, :, i_lm], P8n[:, :, :, i_ln, :], m_max, len_mu, npol, dphi))
                    )

        I_mu = (sum_c + sum_d) @ I_l

        return I_mu


def decompose_ft_phase(ft_p, m_max, len_mu, npol):
    mode = np.arange(0, m_max)
    n_mu = np.arange(0, len_mu)
    # cosines
    ft_p_c = np.array(
        [
            [
                [
                    [ft_p[0, 0, m, n], ft_p[0, 1, m, n], 0],
                    [ft_p[1, 0, m, n], ft_p[1, 1, m, n], 0],
                    [0, 0, ft_p[2, 2, m, n]],
                ]
                for m in mode
            ]
            for n in n_mu
        ]
    )

    # sines
    ft_p_s = np.array(
        [[[[0, 0, -ft_p[0, 2, m, n]], [0, 0, -ft_p[1, 2, m, n]], [ft_p[2, 0, m, n], ft_p[2, 1, m, n], 0]] for m in mode] for n in n_mu]
    )

    # sine = 0 for mode 0
    ft_p_s[:, 0, ] = np.zeros((len_mu, npol, npol))

    return ft_p_c, ft_p_s

def compute_int_phi(mat1, mat2, m_max, len_mu, npol, dphi):
    # approximation of the integral of phi
    # compute the summation of all the mode between two decomposed matrix
    # integral of phi (0, 2pi) using fourier decomposition Appendix 2 tsang et al 2007

    m1_c, m1_s = decompose_ft_phase(mat1, m_max, len_mu, npol)
    m2_c, m2_s = decompose_ft_phase(mat2, m_max, len_mu, npol)

    # error in fourier expansion of tsang? should be mode 0 for second matrix?
    # reshape to match shape of decompose array (len_mu, npol, npol)
    m1_0 = np.array([mat1[:, :, 0, i] for i in range(len_mu)])
    m2_0 = np.array([mat2[:, :, 0, i] for i in range(len_mu)])

    int_0 = 2 * np.pi * (m1_0 @ m2_0)
    sum_mc = 0
    # sum_ms = 0
    # summation of m=1 to m_max, skip 0
    for m in range(m_max)[1:]:
        sum_mc += (m1_c[:, m] @ m2_c[:, m] - m1_s[:, m] @ m2_s[:, m]) * np.cos(m * dphi)
        # sum_ms += (m1_c[:,m] @ m2_s[:,m]  + m1_s[:,m] @ m2_c[:,m]) * np.sin(m * self.dphi) # equal to 0

    int_phi = int_0 + np.pi * sum_mc  # + np.pi * sum_ms

    return int_phi

def get_substrate_reflectivity_integral(substrate, frequency, eps_l, mu_i, mu_int, m_max, npol):
    # Compute the reflectivity matrix need for integrals
    # Need full bi-static to work... doesnt work for iem and geo optics backscatter
    Rbottom_diff_i_int = (substrate.ft_even_diffuse_reflection_matrix(frequency, eps_l, mu_i, mu_int, m_max, npol))
    Rbottom_diff_int_i = (substrate.ft_even_diffuse_reflection_matrix(frequency, eps_l, mu_int, mu_i, m_max, npol))


    # get the two integral, for incident to int (streams) and int to incident)
    Rbottom_diff_int = {'i_int' : Rbottom_diff_i_int,
                        'int_i' : Rbottom_diff_int_i}
    return Rbottom_diff_int


def compute_A(mu_i, mu_int, ke, layer_optical_depth):
    # function of mu and mu'
    mu_i = mu_i[:, np.newaxis, np.newaxis]
    #mu_i = mu_i.reshape((-1, 1, 1))
    gamma_i = compute_gamma(mu_i, layer_optical_depth)
    gamma_int = compute_gamma(mu_int, layer_optical_depth)

    #eqn A11a Karam et al 1995 (everything else except attenuatuin (Sn), phase function and integral)
    A = 1/mu_i * (gamma_i * ((gamma_i - gamma_int) / (ke * (1/mu_i - 1/mu_int)) + 
                                mu_i / (2*ke) * (1 - gamma_i**2))) * mu_i / (ke * (mu_i + mu_int))
    return A

def compute_B(mu_i, mu_int, ke, layer_optical_depth):
    # function of mu and mu'
    mu_i = mu_i[:, np.newaxis, np.newaxis]
    #mu_i = mu_i.reshape((-1, 1, 1))
    gamma_i = compute_gamma(mu_i, layer_optical_depth)
    gamma_int = compute_gamma(mu_int, layer_optical_depth)

    #eqn A11b Karam et al 1995 (everything else except attenuatuin (Sn), phase function and integral)
    B = 1 / (ke * (mu_int + mu_i)) * ((mu_i * (1 - gamma_i**2))/(2*ke) + 
                                        gamma_i * (gamma_int - gamma_i)/(ke * (1/mu_int - 1/mu_i)))
    return B

def compute_C(mu_i, mu_int, ke_n, ke_m, layer_optical_depth_n, layer_optical_depth_m, layer_optical_depth_lr):
    # function of mu and mu'
    mu_i = mu_i[:, np.newaxis, np.newaxis]
    #mu_i = mu_i.reshape((-1, 1, 1))

    gamma_i_n = compute_gamma(mu_i, layer_optical_depth_n)
    gamma_i_m = compute_gamma(mu_i, layer_optical_depth_m)
    gamma_int_n = compute_gamma(mu_int, layer_optical_depth_n)
    gamma_int_m = compute_gamma(mu_int, layer_optical_depth_m)

    #eqn A13a Karam et al 1995 (everything else except phase function and integral)
    C = gamma_int_n * (1 - gamma_i_n * gamma_int_n) / (ke_n * (mu_int + mu_i)) * (gamma_int_m - gamma_i_m) / (ke_m * (1/mu_int - 1/mu_i))
    attenuation_lr = compute_gamma(mu_i, layer_optical_depth_lr) * compute_gamma(mu_int, layer_optical_depth_lr) 

    return C * attenuation_lr


def compute_D(mu_i, mu_int, ke_n, ke_m, layer_optical_depth_n, layer_optical_depth_m, layer_optical_depth_lr):
    # function of mu and mu'
    mu_i = mu_i[:, np.newaxis, np.newaxis]
    #mu_i = mu_i.reshape((-1, 1, 1))

    gamma_i_n = compute_gamma(mu_i, layer_optical_depth_n)
    gamma_i_m = compute_gamma(mu_i, layer_optical_depth_m)
    gamma_int_n = compute_gamma(mu_int, layer_optical_depth_n)
    gamma_int_m = compute_gamma(mu_int, layer_optical_depth_m)

    #eqn A13b Karam et al 1995 (everything else except phase function and integral)
    D = (gamma_i_m - gamma_int_m) / (ke_m * (mu_i - mu_int)) * gamma_i_m * (1 - gamma_int_n * gamma_i_n) / (ke_n * (1/mu_int - 1/mu_i))
    attenuation_lr = compute_gamma(mu_i, layer_optical_depth_lr) * compute_gamma(mu_int, layer_optical_depth_lr) 

    return D * attenuation_lr

def compute_E(mu_i, mu_int, ke, layer_optical_depth, layer_optical_depth_ln_ground):
    # function of mu and mu'
    mu_i = mu_i[:, np.newaxis, np.newaxis]
    gamma_i = compute_gamma(mu_i, layer_optical_depth)
    gamma_int = compute_gamma(mu_int, layer_optical_depth)

    #eqn A8a Karam et al 1995 (everything else except attenuation (Sn), phase function and integral)
    E = mu_i * (gamma_int - gamma_i) / (ke * (mu_int - mu_i))
    #attenuation of layer n to the ground
    attenuation_ln_ground = compute_gamma(mu_i, layer_optical_depth_ln_ground) * compute_gamma(mu_int, layer_optical_depth_ln_ground) 
    
    return E * attenuation_ln_ground 
