"""
Provide the Successive Order Solver as a multi-stream solver of the radiative transfer model based on Lenoble et al.
2007 and Greenwald et al. 2005, with some adaptation using SMRT DORT code.

The main interests of this solver are:

- to provide the succesive orders of interaction separately, allowing the investigation of the
interaction mechanisms.

- faster computations in some conditions: The favorable conditions for fast computation:
    1) shallow snowpack
    2) small grains / weak scattering (i.e. small optical depth)
    3) truncation of the computation at low orders.
    Benchmarking is necessary as DORT can also be orders of magnitude faster for thick snowpacks with big grains.

- an alternative to DORT when it fails due to eigenvalue solution and grains are not very big (scattering is small).


Usage:
    Basic usage with default settings and iba emmodel:
        >>> m = make_model("iba", "successing_order")

References:
    Lenoble, J., Herman, M., Deuzé, J. L., Lafrance, B., Santer, R., & Tanré, D. (2007). A successive order of
    scattering code for solving the vector equation of transfer in the earth’s atmosphere with aerosols. Journal of
    Quantitative Spectroscopy and Radiative Transfer, 107(3), 479–507. https://doi.org/10.1016/j.jqsrt.2007.03.010

    Greenwald, T., Bennartz, R., O’Dell, C., & Heidinger, A. (2005). Fast Computation of Microwave Radiances for Data
    Assimilation Using the “Successive Order of Scattering” Method. Journal of Applied Meteorology, 44(6), 960–966.
    https://doi.org/10.1175/jam2239.1

    Heidinger, A. K., O’Dell, C., Bennartz, R., & Greenwald, T. (2006). The Successive-Order-of-Interaction Radiative
    Transfer Model. Part I: Model Development. Journal of Applied Meteorology and Climatology, 45(10), 1388–1402.
    https://doi.org/10.1175/jam2387.1
"""

# Stdlib import
from collections import defaultdict

# other import
import numpy as np
import numba

# local import
from smrt.core.error import SMRTError, smrt_warn
from smrt.rtsolver.dort import (
    InterfaceProperties,
    symmetrize_phase_matrix,
    matmul,
)  # TODO: move these objects in a generic place
from smrt.rtsolver.rtsolver_utils import RTSolverBase, CoherentLayerMixin, DiscreteOrdinatesMixin

# Lazy import: from smrt.interface.coherent_flat import process_coherent_layers


class SuccessiveOrder(CoherentLayerMixin, DiscreteOrdinatesMixin, RTSolverBase):
    """
    Implement the Successive Order solver.

    Args:
        n_iteration_max: maximum number of computed orders. Setting a value of e.g. 2 only computes first and second
            order scattering. The first order includes only one volume scattering event or one interface reflection.
            Second order includes two volume scattering events, two interface reflections, or one of each.
        relative_tolerance: stop iterating when order[n] = relative_tolerance * order[0].
        n_max_stream: number of stream in the most refringent layer.
        m_max: number of mode (azimuth).
        stream_mode: If set to "most_refringent" (the default) or "air", streams are calculated using the Gauss-Legendre polynomials and
            then use Snell-law to prograpate the direction in the other layers. If set to "uniform_air", streams are calculated
            uniformly in air and then according to Snells law.
        phase_symmetrization: enforce phase function symmetry by replacing the phase function P by (P + P.T)/2 (simplified).
        error_handling: If set to "exception" (the default), raise an exception in case of error, stopping the code.
            If set to "nan", return a nan, so the calculation can continue, but the result is of course unusuable and
            the error message is not accessible. This is only recommended for long simulations that sometimes produce an error.
        process_coherent_layers: Adapt the layers thiner than the wavelegnth using the MEMLS method. The radiative transfer
            theory is inadequate layers thiner than the wavelength and using DORT with thin layers is generally not recommended.
            In some parcticular cases (such as ice lenses) where the thin layer is isolated between large layers, it is possible
            to replace the thin layer by an equivalent reflective interface. This neglects scattering in the thin layer,
            which is acceptable in most case, because the layer is thin. To use this option and more generally
            to investigate ice lenses, it is recommended to read MEMLS documentation on this topic.
    """

    # this specifies which dimension this solver is able to deal with. Those not in this list must be managed by the called (Model object)
    # e.g. here, frequency, time, ... are not managed
    _broadcast_capability = {"theta_inc", "polarization_inc", "theta", "phi", "polarization"}

    def __init__(
        self,
        n_max_stream=32,
        n_iteration_max=50,
        relative_tolerance=0.001,
        m_max=2,
        stream_mode="most_refringent",
        # phase_normalization="auto",
        phase_symmetrization=False,
        error_handling="exception",
        process_coherent_layers=False,
        incident_polarizations="VH",
        # prune_deep_snowpack=None,
    ):
        super().__init__()  # the parent class and mixin must not declare __init__ with parameters

        DiscreteOrdinatesMixin.init(self, n_max_stream=n_max_stream, stream_mode=stream_mode, m_max=m_max)
        CoherentLayerMixin.init(self, process_coherent_layers=process_coherent_layers)

        # self.phase_normalization = phase_normalization
        self.phase_symmetrization = phase_symmetrization
        self.error_handling = error_handling

        if self.phase_symmetrization:
            smrt_warn("symmetrization is under development and it is not sure it is working yet.")

        # if prune_deep_snowpack is True:
        #     prune_deep_snowpack = 6
        # self.prune_deep_snowpack = prune_deep_snowpack

        self.n_iteration_max = n_iteration_max

        if incident_polarizations not in ["V", "VH", "VHU"]:
            raise SMRTError(
                "The argument incident_polarizations must be V, VH or VHU. Note that H only is not supported yet."
            )
        self.incident_polarizations = incident_polarizations

        self.relative_tolerance = relative_tolerance

    def check_sensor(self):
        try:
            len(self.sensor.phi)
        except TypeError:
            pass
        else:
            if len(self.sensor.phi) > 1:
                raise SMRTError("phi as an array must be implemented")

    def solve(self, snowpack, emmodels, sensor, atmosphere=None):
        """Solve the radiative transfer equation for a given snowpack, emmodels and sensor configuration.

        Args:
            snowpack: Snowpack object, :py:mod:`smrt.core.snowpack`.
            emmodels: List of electromagnetic models object, :py:mod:`smrt.emmodel`.
            sensor: Sensor object, :py:mod:`smrt.core.sensor`.
            atmosphere: [Optional] Atmosphere object, :py:mod:`smrt.atmosphere`.

        Returns:
            result: Result object, :py:mod:`smrt.core.result.Result`.
        """

        self.init_solve(snowpack, emmodels, sensor, atmosphere)

        self.process_coherent_layers()  # must be before prepare_streams

        self.prepare_streams()

        m_max = self.m_max if self.sensor.mode == "A" else 0

        # solve the RT equation
        outmu, intensity = self.successing_order(m_max=m_max)

        # interpolate to the requested streams
        intensity = self.interpolate_intensity(outmu, intensity)

        other_coords = [("order", list(range(0, self.n_iteration_max)) + ["total"])]

        return self.make_result(outmu, intensity, other_coords=other_coords)

    def successing_order(self, m_max=0):
        # """Solve the radiative transfer equation using the adding doubling method
        # not to be called by the user
        #     """
        #     :param incident_intensity: give either the intensity (array of size 2) at incident_angle (radar) or isotropic or a function
        #             returning the intensity as a function of the cosine of the angle.
        #     :param incident_angle: if None, the spectrum is considered isotropic, otherise the angle (in radian) given the direction of
        #             the incident beam
        #     :param viewing_phi: viewing azimuth angle, the incident beam is at 0, so pi is the backscatter
        # """

        npol = 3 if self.sensor.mode == "A" else 2

        #
        #   compute the cosine of the angles in all layers
        # first compute the permittivity of the ground

        # prepare the atmosphere
        self.atmosphere_result = (
            self.atmosphere.run(self.sensor.frequency, self.streams.outmu, npol)
            if self.atmosphere is not None
            else None
        )

        # prepare the layer properties
        n_sublayer, extinction, weighted_phase, source = self.prepare_snowpack_properties(m_max)

        #
        # compute interface reflection and transmittance properties
        # prepare the angle of the incident angles requested by the user
        (
            reflection_top,
            reflection_bottom,
            transmission_top,
            transmission_bottom,
            coherent_reflection_top,
            coherent_reflection_bottom,
            coherent_transmission_top,
            coherent_transmission_bottom,
        ) = self.prepare_interfaces(self.streams, m_max)

        # print(f"{n_sublayer=}")
        # add one sub-interface more than sublayer in each layer
        n_subinterface = np.array(n_sublayer) + 1
        i_subinterface = np.insert(np.cumsum(n_subinterface), 0, 0)

        # start iterating
        n = npol * self.streams.n_air
        # print(f"{self.streams.n_air=}")

        if self.sensor.mode == "P":
            intensity_up = np.zeros((npol, self.streams.n_air, self.n_iteration_max))
            profile_shape = (i_subinterface[-1], 2 * npol * max(self.streams.n))
            incident_intensity_0 = None  # TODO: add the tbdown atmopshere here !!!

        elif self.sensor.mode == "A":
            incident_streams = self.prepare_incident_streams()

            incident_npol = len(self.incident_polarizations)

            # print("Number of incident streams:", incident_npol * len(incident_streams))
            intensity_up = np.zeros(
                (npol, self.streams.n_air, incident_npol, len(incident_streams), self.n_iteration_max)
            )
            profile_shape = (i_subinterface[-1], 2 * npol * max(self.streams.n), incident_npol * len(incident_streams))

            # prepare the incident intensity
            incident_intensity_0 = np.empty((n, profile_shape[-1]))
            j = 0
            for i in incident_streams:
                for ipol in range(incident_npol):
                    # set the ieme value at j=incident_streams[i]
                    power = 1.0 / (2 * np.pi * self.streams.outweight[i])
                    incident_intensity_0[npol * i + ipol, j] = power
                    j += 1

            # compute the coherent wave propagation and scattering
            intensity_profile = np.zeros(profile_shape)
            nophase = [np.zeros_like(p) for p in weighted_phase[0]]

            assert self.n_iteration_max > 0
            tolerance = 0  # to be intialized later

            coherent_intensity_up = np.zeros((n, profile_shape[-1], self.n_iteration_max))
            for i in range(self.n_iteration_max):
                intensity_profile, i_up = self.compute_next_order(
                    i,
                    intensity_profile,
                    self.streams.mu,
                    i_subinterface,
                    extinction,
                    source,
                    nophase,  # no scattering
                    coherent_reflection_top,
                    coherent_reflection_bottom,
                    coherent_transmission_top,
                    coherent_transmission_bottom,
                    incident_intensity_0,
                )
                coherent_intensity_up[..., i] = i_up[0:n]

                max_intensity = np.max(i_up[0:n])
                ## print(f"{max_intensity=}")

                if tolerance == 0:
                    # initial the tolerance
                    tolerance = self.relative_tolerance * max_intensity

                # if (max_intensity == 0) or (max_intensity < tolerance):
                #     break

        else:
            raise RuntimeError("unknown sensor mode")

        tolerance = 0  # to be intialized later
        # loop over the mode
        for m in range(m_max + 1):
            # print(f"{m=}")
            # prepare the intensity profile
            intensity_profile = np.zeros(profile_shape)

            if self.sensor.mode == "P":
                intensity_up_m = np.zeros((n, self.n_iteration_max))
            elif self.sensor.mode == "A":
                intensity_up_m = np.zeros_like(coherent_intensity_up)
            else:
                raise RuntimeError("unknown sensor mode")

            incident_intensity = (1 + float(m > 0)) * incident_intensity_0 if incident_intensity_0 is not None else None

            # loop over the sucessive order
            for i in range(self.n_iteration_max):
                intensity_profile, i_up = self.compute_next_order(
                    i,
                    intensity_profile,
                    self.streams.mu,
                    i_subinterface,
                    extinction,
                    source,
                    weighted_phase[m],
                    reflection_top[m],
                    reflection_bottom[m],
                    transmission_top[m],
                    transmission_bottom[m],
                    incident_intensity,
                )
                intensity_up_m[..., i] = i_up[0:n]

                max_intensity = np.max(i_up[0:n])

                if tolerance == 0:
                    # initial the tolerance
                    tolerance = self.relative_tolerance * max_intensity

                if max_intensity < tolerance:
                    break
                # print(
                #    f"Iteration {i}, max intensity={np.max(intensity_profile)}  mean intensity={np.mean(intensity_profile)}"
                # )
            # substract the coherent contribution
            if self.sensor.mode == "A":
                # for i in range(self.n_iteration_max):
                #     print(i, np.mean(intensity_up_m[..., i]), np.mean(coherent_intensity_up[..., i]))
                # # intensity_up_m *= 0
                # intensity_up_m += coherent_i_up[..., np.newaxis] * (1 + float(m > 0))
                # intensity_up_m -= coherent_i_up[..., np.newaxis] * (1 + float(m > 0))
                intensity_up_m -= coherent_intensity_up * (1 + float(m > 0))
                # for i in range(self.n_iteration_max):
                #     print(i, np.mean(intensity_up_m[..., i]))

            if np.ndim(intensity_up_m) == 2:
                intensity_up_m = intensity_up_m.reshape((intensity_up_m.shape[0] // npol, npol, self.n_iteration_max))
                intensity_up_m = np.swapaxes(intensity_up_m, 0, 1)
            elif np.ndim(intensity_up_m) == 3:
                # split the incoming and outgoing polarization:
                intensity_up_m = intensity_up_m.reshape(
                    (
                        intensity_up_m.shape[0] // npol,
                        npol,
                        intensity_up_m.shape[1] // incident_npol,
                        incident_npol,
                        self.n_iteration_max,
                    )
                )
                intensity_up_m = np.swapaxes(np.swapaxes(intensity_up_m, 0, 1), 2, 3)

                # in practice we don't need to substract coherent_i_up if m_max is even, and if only backscatter is
                # needed, considering that each even number cancel each other. However, it is better to keep it here,
                # because we may be interested in bistatic scattering (in other azimuth than np.pi)

            self.add_intensity_mode(intensity_up, intensity_up_m, m=m)

        # reshape the intensity_up npol, incidence, or npol, incidence, npol, incidence

        if self.sensor.mode == "P":
            if self.atmosphere_result is not None:
                intensity_up = self.atmosphere_result.tb_up + self.atmosphere_result.transmittance * intensity_up
            outmu = self.streams.outmu
        elif self.sensor.mode == "A":
            # compress to get only the backscatter
            backscatter_intensity_up = np.zeros((npol, npol, len(incident_streams), self.n_iteration_max))
            for j, i in enumerate(incident_streams):
                # the j-th column vector contains the stream i, with angle mu[i]
                # at the same time, convert the dimension to pola, pola, incidence
                backscatter_intensity_up[:, :incident_npol, j] = intensity_up[:, i, :, j]

            outmu = self.streams.outmu[incident_streams]
            intensity_up = backscatter_intensity_up
        else:
            raise RuntimeError("unknown sensor mode")

        # reshape the first dimension in two dimensions (theta, pola)

        total_intensity_up = np.sum(intensity_up, axis=-1)

        intensity_up = np.append(intensity_up, np.expand_dims(total_intensity_up, -1), axis=-1)

        return outmu, intensity_up

    def prepare_interfaces(self, streams, m_max):
        npol = 3 if self.sensor.mode == "A" else 2

        interfaces = InterfaceProperties(
            self.sensor.frequency,
            self.snowpack.interfaces,
            self.snowpack.substrate,
            self.effective_permittivity,
            streams,
            m_max,
            npol,
        )
        reflection_top = defaultdict(dict)
        reflection_bottom = defaultdict(dict)
        transmission_top = defaultdict(dict)
        transmission_bottom = defaultdict(dict)

        coherent_reflection_top = dict()
        coherent_reflection_bottom = dict()
        coherent_transmission_top = dict()
        coherent_transmission_bottom = dict()

        kwargs = dict(auto_reduce_npol=False)
        for l in range(0, len(streams.mu)):
            for m in range(m_max + 1):
                reflection_top[m][l] = interfaces.reflection_top(l, m=m, compute_coherent_only=False, **kwargs)
                transmission_top[m][l] = interfaces.transmission_top(l, m=m, compute_coherent_only=False, **kwargs)
            coherent_reflection_top[l] = interfaces.reflection_top(l, m=0, compute_coherent_only=True, **kwargs)
            coherent_transmission_top[l] = interfaces.transmission_top(l, m=0, compute_coherent_only=True, **kwargs)
        for l in range(-1, len(streams.mu)):
            for m in range(m_max + 1):
                reflection_bottom[m][l] = interfaces.reflection_bottom(l, m=m, compute_coherent_only=False, **kwargs)
                transmission_bottom[m][l] = interfaces.transmission_bottom(
                    l, m=m, compute_coherent_only=False, **kwargs
                )
            coherent_reflection_bottom[l] = interfaces.reflection_bottom(l, m=0, compute_coherent_only=True, **kwargs)
            coherent_transmission_bottom[l] = interfaces.transmission_bottom(
                l, m=0, compute_coherent_only=True, **kwargs
            )

        return (
            reflection_top,
            reflection_bottom,
            transmission_top,
            transmission_bottom,
            coherent_reflection_top,
            coherent_reflection_bottom,
            coherent_transmission_top,
            coherent_transmission_bottom,
        )

    def prepare_snowpack_properties(self, m_max):
        """
        Args:
        """
        extinction = []
        weighted_phase = list_of_empty_list(m_max + 1)
        n_sublayer = []
        source = []

        for l in range(len(self.emmodels)):
            n, e, wp, s = self.prepare_layer_properties(
                self.snowpack.layers[l], self.emmodels[l], self.streams.mu[l], self.streams.weight[l], m_max
            )

            n_sublayer.append(n)
            extinction.append(e)
            source.append(s)
            for m in range(len(wp)):
                weighted_phase[m].append(wp[m])

        return n_sublayer, extinction, weighted_phase, source

    def prepare_layer_properties(self, layer, emmodel, mu, weight, m_max, infinitesimal_optical_depth=0.1):
        npol = 3 if self.sensor.mode == "A" else 2

        ke = emmodel.ke(mu, npol=npol).compress().diagonal()
        layer_optical_depth = ke * layer.thickness

        n_sublayer = max(int(np.ceil(np.max(layer_optical_depth) / infinitesimal_optical_depth)), 1)

        # compute the phase function
        fullmu = np.concatenate((mu, -mu))
        phase = emmodel.ft_even_phase(mu_s=fullmu, mu_i=fullmu, npol=npol, m_max=self.m_max)

        # apply the factor 2 * pi / 4 * pi. The former is the phi integration, the later is the SMRT convention of the phase function

        full_weight = np.tile(np.repeat(weight, npol), 2)

        extended_mu = np.repeat(mu, npol)
        invke = 1 / ke

        # compute wieghted_phase for each mode
        weighted_phase = []
        for m in range(m_max + 1):
            p = phase.compress(mode=m)
            if self.phase_symmetrization:
                p = symmetrize_phase_matrix(p, m=0)

            coef = 0.5 if m == 0 else 0.25  # coef to apply on the phase matrix, see eq 2 in Greenwald 2005 for m=0
            p *= coef

            weighted_phase.append(np.tile(invke, 2)[:, np.newaxis] * p * full_weight[np.newaxis, :])

        # for active microwave
        # source = single_scattering_albedo * np.exp() * phase[0:n, :] @ intensity0 # eq 31c in Lenoble

        # compute extinction
        extinction = np.exp(-layer_optical_depth / n_sublayer / extended_mu)

        if self.sensor.mode == "P":
            # for passive microwave
            # compute the source
            single_scattering_albedo = emmodel.ks(mu, npol=npol).compress().diagonal() * invke
            source = (1 - single_scattering_albedo) * layer.temperature
        else:
            # for radar microwave
            extinction = extinction[:, np.newaxis]
            source = np.zeros_like(extinction)

        return n_sublayer, extinction, weighted_phase, source

    def compute_next_order(
        self,
        order,
        intensity,
        mu,
        i_subinterface,
        extinction,
        source,
        weighted_phase,
        reflection_top,
        reflection_bottom,
        transmission_top,
        transmission_bottom,
        incident_intensity=None,
    ) -> np.ndarray:
        npol = 3 if self.sensor.mode == "A" else 2

        # intensity is defined at the top of the layer
        new_intensity = np.zeros_like(intensity)

        if (order == 0) and (incident_intensity is not None):
            transmitted_intensity = matmul(transmission_bottom[-1], incident_intensity)
        else:
            transmitted_intensity = None

        # compute the mean_intensity (middle of the layers)
        mean_intensity = (intensity[0:-1, :] + intensity[1:, :]) / 2

        # compute downwelling intensity
        n_layer = len(mu)

        ein_string = "pq, kq -> kp" if self.sensor.mode == "P" else "pq, kqi -> kpi"

        # power_extinction = [extinction[l][np.newaxis, :, :]**np.arange(i_subinterface[l + 1] - i_subinterface[l], 0, -1, dtype=np.float64)[:, np.newaxis, np.newaxis] for l in range(len(extinction))]

        for l in range(n_layer):
            n = npol * len(mu[l])

            # angle and pola
            p_up = slice(0, n)
            p_dn = slice(n, 2 * n)
            q = slice(0, 2 * n)

            # layer slice
            i_top = i_subinterface[l]
            i_bottom = i_subinterface[l + 1] - 1

            new_intensity[i_top, p_dn] = matmul(
                reflection_top[l], intensity[i_top, p_up]
            )  # reflect intensity coming up

            if transmitted_intensity is not None:
                minn = min(n, len(transmitted_intensity))
                new_intensity[i_top, p_dn][0:minn] += transmitted_intensity[0:minn]

            # compute the contribution of scattering of the previous order intensity, which is now a source
            s = np.einsum(
                ein_string,
                weighted_phase[l][p_dn, q],
                mean_intensity[i_top:i_bottom, q],
                optimize=True,
            )

            if order == 0:
                s += source[l]

            s *= 1 - extinction[l]

            series_downwelling(new_intensity[i_top : i_bottom + 1, p_dn], extinction[l], s)

            # for k in range(i_top, i_bottom):
            #     # compute intensity in all the sublayers
            #     new_intensity[k + 1, p_dn] = new_intensity[k, p_dn] * extinction[l] + s[k - i_top]
            #     # eq 66 (adapted for downward)
            # assert k + 1 == i_bottom

            transmitted_intensity = matmul(transmission_bottom[l], new_intensity[i_bottom, p_dn])

        assert i_bottom + 1 == len(new_intensity)

        transmitted_intensity = None

        # compute the upwelling intensity
        for l in range(n_layer - 1, -1, -1):
            n = npol * len(mu[l])

            # angle and pola
            p_up = slice(0, n)
            p_dn = slice(n, 2 * n)
            q = slice(0, 2 * n)

            # layer slice
            i_top = i_subinterface[l]
            i_bottom = i_subinterface[l + 1] - 1

            new_intensity[i_bottom, p_up] = matmul(reflection_bottom[l], intensity[i_bottom, p_dn])
            # reflect intensity coming down at the bottom of the layer

            if transmitted_intensity is not None:
                minn = min(n, len(transmitted_intensity))
                new_intensity[i_bottom, p_up][0:minn] += transmitted_intensity[0:minn]

            s = np.einsum(
                ein_string,
                weighted_phase[l][p_up, q],
                mean_intensity[i_top:i_bottom, q],
                optimize=True,
            )

            if order == 0:
                s += source[l]

            s *= 1 - extinction[l]
            assert len(s) == i_bottom - 1 - i_top + 1

            # for k in range(i_bottom - 1, i_top - 1, -1):
            #     new_intensity[k, p_up] = new_intensity[k + 1, p_up] * extinction[l] + s[k - i_top]
            # assert k == i_top

            series_upwelling(new_intensity[i_top : i_bottom + 1, p_up], extinction[l], s)
            # eq 66 in Lenoble

            transmitted_intensity = matmul(transmission_top[l], new_intensity[i_top, p_up])

        assert i_top == 0
        # compute the final transmission
        emerging_intensity = matmul(transmission_top[0], new_intensity[i_top, p_up])

        # print(f"{np.all(emerging_intensity==0)=} {np.max(emerging_intensity)=} {np.min(emerging_intensity)=}")
        if (incident_intensity is not None) and (order == 0):
            n = len(incident_intensity)
            emerging_intensity[:n] += matmul(reflection_bottom[-1], incident_intensity)

        return new_intensity, emerging_intensity


def list_of_empty_list(n: int):
    return [[] for _ in range(n)]


@numba.jit(nopython=True, cache=True)
def series_downwelling(x, e, s):
    for k in range(len(x) - 1):
        # compute intensity in all the sublayers
        x[k + 1, :] = x[k, :] * e + s[k]


@numba.jit(nopython=True, cache=True)
def series_upwelling(x, e, s):
    for k in range(len(x) - 2, -1, -1):
        # compute intensity in all the sublayers
        x[k, :] = x[k + 1, :] * e + s[k]


# @numba.jit(nopython=True, cache=True)
# def series_upwelling(x, e, s):

#     for k in range(len(x) - 2, -1, -1):
#         for i in range(x.shape[1]):
#             # compute intensity in all the sublayers
#             x[k, i] = x[k + 1, i] * e[i] + s[k, i]

# def series_downwelling_cumsum(x, power_e, s):
#     #print(x.shape, power_e.shape)
#     # power_e = e[np.newaxis, :, :]**np.arange(len(x), 0, -1)[:, np.newaxis, np.newaxis]
#     x[1:] = np.cumsum(power_e[:-1] * s, axis=0) / power_e[:-1] + x[0] * power_e[-1:0:-1]
#     return x

# from numba import cuda
# import math
# # start code for using cuda
# #change grid(2) to grid 1, and consider the organisation of the *input x [k, i, j]

# def series_downwelling_gpu(x, e, s):
#     print(x.shape, e.shape, s.shape)

#     block_dim_i = 2
#     block_dim_j = 2
#     n_block_i = math.ceil(x.shape[1] / block_dim_i)
#     n_block_j = math.ceil(x.shape[2] / block_dim_j)
#     series_downwelling_cuda[(n_block_i, n_block_j), (block_dim_i, block_dim_j)](x, e, s)


# @cuda.jit  # !!! install the correct version: https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html
# def series_downwelling_cuda(x, e, s):
#     # Get the 2D thread indices (only i, j are part of the grid)
#     i, j = cuda.grid(2)

#     if i >= x.shape[1]:
#         return
#     if j >= x.shape[2]:
#         return

#     # Loop over k to compute x[k+1] sequentially
#     for k in range(x.shape[0] - 1):
#         x[k + 1, i, j] = e[i, 1] * x[k, i, j] + s[k, i, j]
