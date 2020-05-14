# coding: utf-8

"""The Discrete Ordinate and Eigenvalue Solver is a multi-stream solver of the radiative transfer model. It is precise but less efficient 
than 2 or 6 flux solvers. Different flavours of DORT (or DISORT) exist depending on the mode (passive or active), on the density of the medium
(sparse media have trivial inter-layer boundary conditions), on the way the streams are connected between the layers and on the way the phase
function is prescribed. The actual version is a blend between Picard et al. 2004 (active mode for sparse media) and DMRT-ML (Picard et al. 2013) which works
in passive mode only for snow. The DISORT often used in optics (Stamnes et al. 1988) works only for sparse medium and uses a development of the phase
function in Legendre polynomia on theta. The version used in DMRT-QMS (L. Tsang's group) is similar to the present implementation except
it uses spline interpolation to connect constant-angle streams between the layers although we use direct connection by varying the angle 
according to Snell's law. A practical consequence is that the number of streams vary (due to internal reflection) and the value `n_max_stream`
only applies in the most refringent layer. The number of outgoing streams in the air is usually smaller, sometimes twice smaller (depends on the density profile).
It is important not to set too low a value for n_max_streams. E.g. 32 is usually fine, 64 or 128 are better but simulations will be much slower.

"""


# Stdlib import
import math
from warnings import warn

# other import
import numpy as np
import scipy.special.orthogonal
import scipy.linalg
import scipy.interpolate

# local import
from ..core.error import SMRTError
from ..core.result import make_result
from smrt.core.lib import smrt_matrix, smrt_diag, isnull
from smrt.core.optional_numba import numba
# Lazy import: from smrt.interface.coherent_flat import process_coherent_layers 


class DORT(object):
    """Discrete Ordinate and Eigenvalue Solver

        :param n_max_stream: number of stream in the most refringent layer
        :param m_max: number of mode (azimuth)
        :param phase_normalization: the integral of the phase matrix should in principe be equal to the scattering coefficient.
        However, some emmodels do not respect this strictly. In general a small difference is due to numerical rounding and is acceptable,
        but a large difference rather indicates either a bug in the emmodel or input parameters that breaks the
        assumption of the emmodel. The most typical case is when the grain size is too big compared to wavelength for emmodels
        that rely on Rayleigh assumption. If this argument is to True (the default), the phase matrix is normalized to be coherent
        with the scattering coefficient, but only when the difference is moderate (0.7 to 1.3).
        If set to "force" the normalization is always performed. This option is dangerous because it may hide bugs or unappropriate
        input parameters (typically too big grains). If set to False, no normalization is performed.
        :param error_handling: If set to "exception" (the default), raise an exception in cause of error, stopping the code. If set to "nan", return a nan, so the calculation can continue, 
        but the result is of course unusuable and the error message is not accessible. This is only recommended for long simulations that sometimes produce an error.
        :param process_coherent_layers: Adapt the layers thiner than the wavelegnth using the MEMLS method. The radiative transfer theory is inadequate
        layers thiner than the wavelength and using DORT with thin layers is generally not recommended. In some parcticular cases (such as ice lenses)
        where the thin layer is isolated between large layers, it is possible to replace the thin layer by an equivalent reflective interface.
        This neglects scattering in the thin layer, which is acceptable in most case, because the layer is thin. To use this option and more generally 
        to investigate ice lenses, it is recommended to read MEMLS documentation on this topic.
        :param prune_deep_snowpack: this value is the optical depth from which the layers are discarded in the calculation. It is to be use to accelerate the calculations
        for deep snowpacks or at high frequencies when the contribution of the lowest layers is neglegible. The optical depth is a good criteria to determine this limit.
        A value of about 6 is recommended. Use with care, especially values lower than 6.
    """

    # this specifies which dimension this solver is able to deal with. Those not in this list must be managed by the called (Model object)
    # e.g. here, frequency, time, ... are not managed
    _broadcast_capability = {"theta_inc", "polarization_inc", "theta", "phi", "polarization"}

    def __init__(self,
                 n_max_stream=32,
                 m_max=2,
                 stream_mode="most_refringent",
                 phase_normalization=True,
                 error_handling="exception",
                 process_coherent_layers=False,
                 prune_deep_snowpack=None):
        # """
        # :param n_max_stream: number of stream in the most refringent layer
        # :param m_max: number of mode (azimuth)

        # """
        self.n_max_stream = n_max_stream
        self.stream_mode = stream_mode
        self.m_max = m_max
        self.phase_normalization = phase_normalization
        self.error_handling = error_handling
        self.process_coherent_layers = process_coherent_layers

        if prune_deep_snowpack is True:
            prune_deep_snowpack = 6
        self.prune_deep_snowpack = prune_deep_snowpack

    def solve(self, snowpack, emmodels, sensor, atmosphere=None):
        """solve the radiative transfer equation for a given snowpack, emmodels and sensor configuration.

"""
        try:
            len(self.sensor.phi)
        except Exception:
            pass
        else:
            raise Exception("phi as an array must be implemented")

        if self.process_coherent_layers:
            from smrt.interface.coherent_flat import process_coherent_layers  # we only import this if requested by the users.
            snowpack, emmodels = process_coherent_layers(snowpack, emmodels, sensor)  

        # all these assignements are for convenience, this would break with parallel code (// !!)
        self.emmodels = emmodels
        self.snowpack = snowpack
        self.sensor = sensor

        self.atmosphere = atmosphere

        self.effective_permittivity = np.array([emmodel.effective_permittivity() for emmodel in emmodels])
        self.substrate_permittivity = self.snowpack.substrate.permittivity(self.sensor.frequency) \
            if self.snowpack.substrate is not None else None

        if self.sensor.mode == 'P':
            pola = ['V', 'H']
            self.temperature = [layer.temperature for layer in self.snowpack.layers]
            m_max = 0  # force m_max=0 for passive microwave
        else:
            pola = ['V', 'H', 'U']
            self.temperature = None
            m_max = self.m_max

        # solve the RT equation
        outmu, intensity = self.dort(m_max=m_max)

        # reshape the first dimension in two dimensions (theta, pola)
        npol = len(pola)
        intensity = intensity.reshape([intensity.shape[0] // npol, npol] + list(intensity.shape[1:]))

        mu = np.cos(sensor.theta)

        fill_value = None
        if np.max(mu) > np.max(outmu):
            # need extrapolation to 0°
            # add the mean of H and V polarisation for the smallest angle for theta=0 (mu=1)
            if self.sensor.mode == 'P':  # passive
                outmu = np.insert(outmu, 0, 1.0)
                intensity = np.insert(intensity, 0, np.mean(intensity[0, :, ...], axis=0), axis=0)
            else:  # active
                copol = (intensity[0, 0, 0] + intensity[0, 1, 1]) / 2
                crosspol = (intensity[0, 1, 0] + intensity[0, 0, 1]) / 2

                intensity = np.insert(intensity, 0, [[copol, crosspol, intensity[0, 0, 2]],
                                                     [crosspol, copol, intensity[0, 1, 2]],
                                                     intensity[0, 2, :]], axis=0)
                outmu = np.insert(outmu, 0, 1.0)

        if np.min(mu) < np.min(outmu):
            raise SMRTError("Viewing zenith angle is higher than the stream angles computed by DORT."
                            " Either increase the number of streams or reduce the highest viewing zenith angle.")

        # reverse is necessary for "old" scipy version
        intfct = scipy.interpolate.interp1d(outmu[::-1], intensity[::-1, ...],
                                            axis=0, fill_value=fill_value, assume_sorted=True)
        # the previous call could use fill_value to be smart about extrapolation, but it's safer to return NaN (default)

        # it seems there is a bug in scipy at least when checking the boundary, mu must be sorted
        # original code that should work: intensity = intfct(mu)
        i = np.argsort(mu)
        intensity = intfct(mu[i])[np.argsort(i)]  # mu[i] sort mu, and [np.argsort(i)] put in back

        #if sensor.mode == 'A':
        #    # reshape the outer/first dimension in two dimensions (theta_inc, pola_inc)
        #    intensity = intensity.reshape(list(intensity.shape[:-1]) + [intensity.shape[-1] // npol, npol])

        #  describe the results list of (dimension name, dimension array of value)
        if sensor.mode == 'P':
            coords = [('theta', sensor.theta_deg), ('polarization', pola)]

        else: # sensor.mode == 'A':
            #coords = [('theta_inc', sensor.theta_inc_deg), ('polarization_inc', pola)] + coords
            coords = [('theta_inc', sensor.theta_inc_deg), ('polarization_inc', pola), ('polarization', pola)]

        return make_result(sensor.mode, intensity, coords)

    def dort(self, m_max=0, special_return=False):
        # not to be called by the user
        #     """
        #     :param incident_intensity: give either the intensity (array of size 2) at incident_angle (radar) or isotropic or a function
        #             returning the intensity as a function of the cosine of the angle.
        #     :param incident_angle: if None, the spectrum is considered isotropic, otherise the angle (in radian) given the direction of
        #             the incident beam
        #     :param viewing_phi: viewing azimuth angle, the incident beam is at 0, so pi is the backscatter
        # """

        #
        #   compute the cosine of the angles in all layers
        # first compute the permittivity of the ground

        streams = compute_stream(self.n_max_stream, self.effective_permittivity, self.substrate_permittivity, mode=self.stream_mode)

        #
        # compute the incident intensity array depending on the sensor

        intensity_0, intensity_higher, incident_streams = self.prepare_intensity_array(streams)  # TODO Ghi: make an iterator

        npol = 3 if self.sensor.mode == 'A' else 2

        #
        # compute interface reflection and transmittance properties

        interfaces = InterfaceProperties(self.sensor.frequency, self.snowpack.interfaces, self.snowpack.substrate,
                                         self.effective_permittivity, streams, m_max, npol)

        #
        # create eigenvalue solvers
        eigenvalue_solver = [EigenValueSolver(self.emmodels[l].ke,
                                              self.emmodels[l].ks,
                                              self.emmodels[l].ft_even_phase,
                                              streams.mu[l],
                                              streams.weight[l],
                                              m_max,
                                              self.phase_normalization) for l in range(len(self.emmodels))]

        #
        # compute the outgoing intensity for each mode

        for m in range(0, m_max + 1):
            intensity_down_m = intensity_0 if m == 0 else intensity_higher

            # compute the upwelling intensity for mode m
            intensity_up_m = self.dort_modem_banded(m, streams, eigenvalue_solver, interfaces, intensity_down_m,
                                                    special_return=special_return)

            if special_return:  # debuging
                return intensity_up_m

            if self.sensor.mode == 'A':
                # substrate the coherent contribution
                intensity_up_m -= self.dort_modem_banded(m, streams, eigenvalue_solver, interfaces, intensity_down_m,
                                                         compute_coherent_only=True)

            # reconstruct the intensity
            if m == 0:
                intensity_up = extend_2pol_npol(intensity_up_m, npol)
            else:
                intensity_up[0::npol] += intensity_up_m[0::npol] * np.cos(m * self.sensor.phi)  # TODO Ghi: deals with an array of phi
                intensity_up[1::npol] += intensity_up_m[1::npol] * np.cos(m * self.sensor.phi)  # TODO Ghi: deals with an array of phi
                intensity_up[2::npol] += intensity_up_m[2::npol] * np.sin(m * self.sensor.phi)  # TODO Ghi: deals with an array of phi

                # TODO: implement a convergence test if we want to avoid long computation
                # when self.m_max is too high for the phase function.

        if self.sensor.mode == 'P' and self.atmosphere is not None:
            intensity_up = self.atmosphere.tbup(self.sensor.frequency, streams.outmu, npol) + \
                self.atmosphere.trans(self.sensor.frequency, streams.outmu, npol) * intensity_up

        if self.sensor.mode == 'A':
            # compress to get only the backscatter
            backscatter_intensity_up = np.empty((npol * len(incident_streams), npol))
            for j, i in enumerate(incident_streams):
                # the j-th column vector contains the stram i, with angle mu[i]
                backscatter_intensity_up[3 * j: 3 * j + 3, :] = intensity_up[3 * i: 3 * i + 3, 3 * j: 3 * j + 3]

            outmu = streams.outmu[incident_streams]
            intensity_up = backscatter_intensity_up
        else:
            outmu = streams.outmu

        return outmu, intensity_up

    def prepare_intensity_array(self, streams):

        if self.sensor.mode == 'A':
            # send a direct beam

            # incident angle at a given angle
            # use interpolation to get the based effective angle

            #
            # delta(x) = 1/2pi + 1/pi*sum_{n=1}{infinty} cos(nx)
            #

            incident_streams = set()

            for theta in self.sensor.theta_inc:
                mu_inc = math.cos(theta)
                i0 = np.searchsorted(-streams.outmu, -mu_inc)
                if i0 == 0:
                    incident_streams.add(i0)
                elif i0 == len(streams.outmu):
                    incident_streams.add(i0 - 1)
                else:
                    incident_streams.add(i0)
                    incident_streams.add(i0 - 1)
            incident_streams = sorted(list(incident_streams))  # fix the order (required for the interpolation)

            intensity_0 = np.zeros((2 * len(streams.outmu), 2 * len(incident_streams)))  # 2 is for the two polarizations
            intensity_higher = np.zeros((3 * len(streams.outmu), 3 * len(incident_streams)))  # 2 is for the two polarizations

            j0 = 0
            j_higher = 0
            for i in incident_streams:
                power = 1.0 / (2 * math.pi * streams.outweight[i])
                for ipol in [0, 1]:
                    intensity_0[2 * i + ipol, j0] = power
                    j0 += 1
                for ipol in [0, 1, 2]:
                    intensity_higher[3 * i + ipol, j_higher] = power
                    j_higher += 1

        elif self.sensor.mode == 'P':

            npol = 2
            incident_streams = []

            if self.atmosphere is not None:

                # incident radiation is a function of frequency and incidence angle
                # assume azimuthally symmetric
                intensity_0 = self.atmosphere.tbdown(self.sensor.frequency, streams.outmu, npol)[:, np.newaxis]
                intensity_higher = np.zeros_like(intensity_0)

            else:
                intensity_0 = np.zeros((len(streams.outmu) * npol, 1))
                intensity_higher = intensity_0
                intensity_0.flags.writeable = False  # make immutable
                intensity_higher.flags.writeable = False  # make immutable

        else:
            raise SMRTError("Unknow sensor mode")

        return intensity_0, intensity_higher, incident_streams

    def dort_modem_banded(self, m, streams, eigenvalue_solver,
                          interfaces, intensity_down_m,
                          compute_coherent_only=False, special_return=False):

        # Index convention
        # for phase, Ke, and R matrix pola must be the fast index, then stream, then +-
        # order of the boundary condition in row: layer, equation, stream+/stream-, pola
        # order of the boundary condition in column: layer, -/+, etc

        npol = 2 if m == 0 else 3

        # indexes of the columns
        jl = 2 * (np.cumsum(streams.n) - streams.n) * npol

        # indexes of the rows: start Eq 19, then Eq TOP then Eq BOTTOM, then Eq 22
        il_top = 2 * (np.cumsum(streams.n) - streams.n) * npol
        il_bottom = il_top + streams.n * npol

        nboundary = sum(streams.n) * 2 * npol
        if len(streams.n) >= 2:
            nband = npol * max(np.max(2 * streams.n[1:] + streams.n[:-1]),
                               np.max(streams.n[1:] + 2 * streams.n[:-1]))
            #print("gain:", nband / (3 * npol * np.max(streams.)))
            # in principle could be better optimized as the number of upper and lower diagol can be different
        else:
            nband = 3 * npol * np.max(streams.n)  # each layer appears in 3 blocks
        # (bottom, top of the current layer, and top of layer below (for downward directons) and
        # bottom of the layer above (for upward directions)

        debug_compute_BC = special_return in ["BC", "testeq"]  # compute the full matrix boundary condition

        # Boundary condition matrix
        bBC = np.zeros((2 * nband + 1, nboundary))  # we use banded Boundary condition matrix

        # rhs vector size
        assert(len(intensity_down_m.shape) == 2)
        nvector = intensity_down_m.shape[1]
        b = np.zeros((nboundary, nvector))

        if debug_compute_BC:
            BC = np.zeros((nboundary, nboundary))  # full/dense Boundary condition matrix. Only for debugging.

        nlayer = len(eigenvalue_solver)

        # used to estimate if the medium is deep enough
        optical_depth = 0

        for l in range(0, nlayer):
            nsl = streams.n[l]  # number of streams in layer l
            nsl_npol = nsl * npol  # number of streams * npol in layer l
            nsl2_npol = 2 * nsl_npol    # number of eigenvalues in layer l (should be 2 * npol*n_stream)
            nslm1_npol = (streams.n[l - 1] * npol) if l > 0 else (streams.n_air * npol)  # number of streams * npol in the layer l - 1 (lm1)
            # number of streams * npol in the layer l + 1 (lp1)
            nslp1_npol = (streams.n[l + 1] * npol) if l < nlayer - 1 else (streams.n_substrate * npol)

            # solve the eigenvalue problem for layer l

            # TODO: the following duplicates the eignevalue_solver call line. A better way should be implemented,
            # either with a variable holding the exception type (
            # and use of a never raised exception or see contextlib module if useful)
            if self.error_handling == 'nan':
                try:
                    # run in a try to catch the exception
                    beta, Eu, Ed = eigenvalue_solver[l].solve(m, compute_coherent_only)
                except SMRTError:
                    return np.full_like(intensity_down_m, np.nan).squeeze()
            else:
                beta, Eu, Ed = eigenvalue_solver[l].solve(m, compute_coherent_only)
            assert(Eu.shape[0] == npol * nsl)

            # deduce the transmittance through the layers
            # positive beta, reference at the bottom
            transt = smrt_diag(np.exp(-np.maximum(beta, 0) * self.snowpack.layers[l].thickness))
            # negative beta, reference at the top
            transb = smrt_diag(np.exp(np.minimum(beta, 0) * self.snowpack.layers[l].thickness))

            # where we have chosen
            # beta>0  : z(0)(l) = z(l)    # reference is at the bottom
            # beta<0  : z(0)(l) = z(l - 1)  # reference is at the top
            # so that the transmittance are < 1

            # few short-cut
            il_topl = il_top[l]  # row of the top boundary condition for layer l
            il_bottoml = il_bottom[l]  # row of the bottom boundary condition for layer l
            j = jl[l]

            # -------------------------------------------------------------------------------
            # Eq 17 & 19 TOP of layer l
            if l == 0:
                # save these matrix to compute the emerging intensity at the end
                Eu_0 = Eu
                transt_0 = transt

            # compute reflection coefficient between l and l - 1
            Rtop_l = interfaces.reflection_top(l, m, compute_coherent_only)

            # fill the matrix
            todiag(bBC, il_topl, j, matmul(Ed - matmul(Rtop_l, Eu), transt))

            if l < nlayer - 1:
                ns_npol_common_bottom_ = min(nsl_npol, nslp1_npol)
                Tbottom_lp1 = interfaces.transmission_bottom(l, m, compute_coherent_only)
                todiag(bBC, il_top[l + 1], j, -matmul(Tbottom_lp1, Ed[0:ns_npol_common_bottom_, :], transb))

            # fill the vector
            if m == 0 and self.temperature is not None and self.temperature[l] > 0:
                if isnull(Rtop_l):
                    b[il_topl:il_topl + nsl_npol, :] -= self.temperature[l]  # a mettre en (l)
                else:
                    b[il_topl:il_topl + nsl_npol, :] -= ((1.0 - muleye(Rtop_l)) * self.temperature[l])[:, np.newaxis]  # a mettre en (l)
                # the muleye comes from the isotropic emission of the black body

                if l < nlayer - 1 and self.temperature[l] > 0:
                    b[il_top[l + 1]:il_top[l + 1] + ns_npol_common_bottom_, :] += \
                        (muleye(Tbottom_lp1) * self.temperature[l])[:, np.newaxis]     # a mettre en (l + 1)

            if l == 0:  # Air-snow interface
                Tbottom_air_down = interfaces.transmission_bottom(-1, m, compute_coherent_only)
                b[il_topl:il_topl + streams.n_air * npol, :] += matmul(Tbottom_air_down, intensity_down_m)

            # -------------------------------------------------------------------------------
            # Eq 18 & 22 BOTTOM of layer l

            # compute reflection coefficient between l and l + 1
            Rbottom_l = interfaces.reflection_bottom(l, m, compute_coherent_only)

            # fill the matrix
            todiag(bBC, il_bottoml, j, matmul(Eu - matmul(Rbottom_l, Ed), transb))

            if l > 0:
                ns_npol_common_top_ = min(nsl_npol, nslm1_npol)
                Ttop_lm1 = interfaces.transmission_top(l, m, compute_coherent_only)
                todiag(bBC, il_bottom[l - 1], j, -matmul(Ttop_lm1, Eu[0:ns_npol_common_top_, :], transt))

            # fill the vector
            if m == 0 and self.temperature is not None and self.temperature[l] > 0:
                if isnull(Rbottom_l):
                    b[il_bottoml:il_bottoml + nsl_npol, :] -= self.temperature[l]   # a mettre en (l)
                else:
                    b[il_bottoml:il_bottoml + nsl_npol, :] -= \
                        ((1.0 - muleye(Rbottom_l)) * self.temperature[l])[:, np.newaxis]  # a mettre en (l)
                if l > 0:
                    b[il_bottom[l - 1]:il_bottom[l - 1] + ns_npol_common_top_, :] += \
                        (muleye(Ttop_lm1) * self.temperature[l])[:, np.newaxis]  # a mettre en (l - 1)

            if m == 0 and l == nlayer - 1 and self.snowpack.substrate is not None and \
                    self.snowpack.substrate.temperature is not None and self.temperature is not None:
                Tbottom_sub = interfaces.transmission_bottom(l, m, compute_coherent_only)
                b[il_bottoml:il_bottoml + nsl_npol, :] += (muleye(Tbottom_sub) * self.snowpack.substrate.temperature)[:, np.newaxis]

            # Finalize
            optical_depth += np.min(np.abs(beta)) * self.snowpack.layers[l].thickness

            if self.prune_deep_snowpack is not None and optical_depth > self.prune_deep_snowpack:
                # prune the matrix and vector
                nboundary = sum(streams.n[0:l + 1]) * 2 * npol
                bBC = bBC[:, 0:nboundary]
                b = b[0:nboundary, :]

                break

        # -------------------------------------------------------------------------------
        #   solve the boundary system BCx=b

        if special_return == "BC":
            return BC, b
        elif special_return == "bBC":
            return bBC, b

        if special_return == "testeq":
            # test
            x = scipy.linalg.solve(BC, b, overwrite_a=True, overwrite_b=False)
            x2 = scipy.linalg.solve_banded((nband, nband), bBC, b, overwrite_ab=True, overwrite_b=False)
            np.testing.assert_allclose(x, x2, rtol=1e-06)
            print("both matrix are equal")

        if self.snowpack.substrate is None and optical_depth < 5:
            warn("DORT has detected that the snowpack is optically shallow (tau=%g)and no substrate has been set, meaning that the space "
                 "under the snowpack is vaccum and that the snowpack is shallow enough to affect the signal measured at the surface."
                 "This is usually not wanted. Either increase the thickness of the snowpack or set a substrate."
                 " If wanted, add a transparent substrate to supress this warning" % optical_depth)

        x = scipy.linalg.solve_banded((nband, nband), bBC, b, overwrite_ab=True, overwrite_b=True)

        # #  ! calculate the intensity emerging from the snowpack
        l = 0
        j = jl[l]  # should be 0
        nsl2_npol = 2 * streams.n[l] * npol
        I1up_m = Eu_0 @ transt_0 @ x[j:j + nsl2_npol, :]

        if m == 0 and self.temperature is not None and self.temperature[0] > 0:
            I1up_m += self.temperature[0]  # just under the interface

        Rbottom_air_down = interfaces.reflection_bottom(-1, m, compute_coherent_only)
        Ttop_0 = interfaces.transmission_top(0, m, compute_coherent_only)  # snow-air
        I0up_m = Rbottom_air_down @ intensity_down_m + (Ttop_0 @ I1up_m[:streams.n_air * npol])

        return np.array(I0up_m).squeeze()


def muleye(x):
    #  """multiply x * 1v """

    if isinstance(x, smrt_diag):
        return x.diagonal()
    else:
        return x


def matmul(a, b, *args):
    # """just because numpy matrix operator does not support scalar multiplication..."""
    if args:
        b = matmul(b, *args)
    if np.isscalar(a) or np.isscalar(b):
        return a * b
    else:
        return a @ b


def todiag(bmat, oi, oj, dmat):
    #"""insert the small dense dmat matrix in the diagonal bmat matrix"""

    u = (bmat.shape[0]-1) // 2
    l = u  # special case here

    n, m = dmat.shape
    dmat_flat = dmat.flatten()

    for j in range(0, m):
        ldiag = min(n, m-j)
        bmat[u+oi-oj-j, j+oj:j+oj+ldiag] = dmat_flat[j:j+(m+1)*ldiag:m+1]

    for i in range(1, n):
        ldiag = min(n-i, m)
        bmat[u+oi-oj+i, 0+oj:0+oj+ldiag] = dmat_flat[i*m:i*m+(m+1)*ldiag:m+1]

if numba:
    compiled_todiag = numba.jit(nopython=True, cache=True)(todiag)
    def todiag(bmat, oi, oj, dmat):
        compiled_todiag(bmat, int(oi), int(oj), dmat)


def extend_2pol_npol(x, npol):

    if npol == 2:
        return x

    if scipy.sparse.isspmatrix_dia(x):
        y = scipy.sparse.diags(extend_2pol_npol(x.diagonal(), npol))
    elif len(x.shape) == 1:
        y = np.zeros(len(x) // 2 * npol)
        y[0::npol] = x[0::2]
        y[1::npol] = x[1::2]
    elif len(x.shape) == 2:
        y = np.zeros((x.shape[0] // 2 * npol, x.shape[1] // 2 * npol))
        y[0::npol, 0::npol] = x[0::2, 0::2]
        y[0::npol, 1::npol] = x[0::2, 1::2]
        y[1::npol, 0::npol] = x[1::2, 0::2]
        y[1::npol, 1::npol] = x[1::2, 1::2]
    else:
        raise SMRTError("should never be here")

    return y


class EigenValueSolver(object):

    def __init__(self, ke, ks, ft_even_phase_function, mu, weight, m_max, normalization):
        # :param Ke: extinction coefficient of the layer for mode m
        # :param ft_even_phase: ft_even_phase function of the layer for mode m
        # :param mu: cosines
        # :param weight: weights

        self.ke = ke
        self.ks = ks
        self.mu = mu
        self.weight = weight
        self.normalization = normalization
        self.norm_0 = None
        self.norm_m = None

        if ft_even_phase_function is not None:
            mu = np.concatenate((self.mu, -self.mu))
            self.ft_even_phase = ft_even_phase_function(mu, mu, m_max)
        else:
            self.ft_even_phase = smrt_matrix(0)

    def solve(self, m, compute_coherent_only):
        # solve the homogeneous equation for a single layer and return eigne value and eigen vector
        # :param m: mode
        # :param compute_coherent_only
        # :returns: beta, E, Q
        #

        npol = 2 if m == 0 else 3

        n = npol * len(self.mu)

        # this coefficient comme from the 1/4pi normalization of the RT equation and the 
        # 1/(4*pi) * int_{phi=0}^{2*pi} cos(m phi)*cos(n phi) dphi
        # note that equation A7 and A8 in Picard et al. 2018 has an error, it does not show this coefficient.
        coef = 0.5 if m == 0 else 0.25

        # compute invmu
        invmu = 1.0 / self.mu
        invmu = np.repeat(invmu, npol)
        invmu = np.concatenate((invmu, -invmu))
        mu = np.concatenate((self.mu, -self.mu))

        # calculate the A matrix. Eq (12),  or 0 if compute_coherent_only
        A = self.ft_even_phase.compress(mode=m, auto_reduce_npol=True) if not compute_coherent_only else 0

        if isnull(A):
            # the solution is trivial
            beta = invmu * np.repeat(self.ke(mu), npol)
            E = np.eye(2 * n, 2 * n)
        else:
            # solve the eigen value problem
            coef_weight = np.tile(np.repeat(-coef * self.weight, npol), 2)    # could be cached (per layer) because same for each mode

            A *= coef_weight[np.newaxis, :]

            # normalize
            if self.normalization and self.ks > 0:
                A = self.normalize(m, A)
            # normalization is done

            A[np.diag_indices(2 * n)] += np.repeat(self.ke(mu), npol)
            A = invmu[:, np.newaxis] * A

            # diagonalise the matrix. Eq (13)
            try:
                beta, E = scipy.linalg.eig(A, overwrite_a=True)
            except scipy.linalg.LinAlgError:
                diagonalization_failed = True
                reason = "eig method"
            else:
                notclose_beta = not np.allclose(beta.imag, 0, atol=1e-06)
                notclose_E = not np.allclose(E.imag, 0, atol=1e-06)
                diagonalization_failed = notclose_beta or notclose_E

                reason = ""
                if notclose_beta:
                    reason += "not close beta "
                if notclose_E:
                    reason += "not close E "

            if diagonalization_failed:
                print("Reason: ", reason, " ks:", self.ks)
                mask = np.abs(E.imag) > 1e-8
                print("Info:", m, E[mask], beta[np.any(mask, axis=0)])
                raise SMRTError("""The diagonalization failed in DORT which is possibly caused by single scattering albedo larger than 1.
It is often due to grain size too large (or too low stickiness parameter) to respect the Rayleigh/low-frequency assumption required by
some emmodel (DMRT ShortRange, IBA, ...). It is recommended to reduce the size of the bigger grains. It is possible to disable this error
raise and return NaN instead by adding the argument rtsolver_options=dict(error_handling='nan') to make_model).
""")

            if np.iscomplexobj(E):
                mask = abs(E.imag) > np.linalg.norm(E) * 1e-5
                if np.any(mask):
                    print(np.any(mask, axis=1))
                    print(beta[np.any(mask, axis=1)])
                    print(beta)

            beta = beta.real

            # get the positive and negative beta
            # this should be improve a the mathematical level because there is no need to solve
            # the eigenvalue for + and - well according to inverse optical path equivalent,
            # the + and - should be equal

            # debug only
            if False:
                idx2 = np.argsort(beta)
                idx2[0:n] = idx2[0:n][::-1]
                beta = beta[idx2]
                E = E[:, idx2]

        Eu = E[0:n, :]  # upwelling
        Ed = E[n:, :]  # downwelling

        return beta, Eu, Ed
        # !-----------------------------------------------------------------------------!

    def normalize(self, m, A):
        # normalize A to conserve energy, assuming isotrope scattering coefficient
        # ks should be a function of mu if non-isotropic medium and to be consistent with ke which is a function of mu

        npol = 2 if m == 0 else 3

        if m == 0:
            self.norm_0 = -self.ks / np.sum(A, axis=1)
            norm = self.norm_0

            if self.normalization != "forced" and np.any(np.abs(self.norm_0 - 1.0) > 0.3):
                raise SMRTError("""The re-normalization of the phase function exceeds the predefined threshold of 30%.
This is likely because of a too large grain size or a bug in the phase function. It is recommended to check the grain size.
You can also deactivate this check using normalization="forced" as an options of the dort solver. It is at last possible
to disable this error raise and return NaN instead by adding the argument rtsolver_options=dict(error_handling='nan') to make_model).""")
        else:
            if self.norm_m is None:
                if self.norm_0 is None:  # be careful, this code is not re-entrant
                    raise Exception("For the normalization, it is necessary to call this function for the mode m=0 first.")
                # transform the norm_0 for npol
                self.norm_m = np.empty(len(self.norm_0) // 2 * npol)
                self.norm_m[0::npol] = self.norm_0[0::2]
                self.norm_m[1::npol] = self.norm_0[1::2]
                for ipol in range(2, npol):
                    # this approach is empirical
                    self.norm_m[ipol::npol] = np.sqrt(self.norm_0[0::2] * self.norm_0[1::2])
            norm = self.norm_m

        A *= norm[:, np.newaxis]
        return A


class InterfaceProperties(object):

    def __init__(self, frequency, interfaces, substrate, permittivity, streams, m_max, npol):

        self.streams = streams

        nlayer = len(interfaces)

        self.Rtop_coh = dict()
        self.Ttop_coh = dict()
        self.Rbottom_coh = dict()
        self.Rbottom_diff = dict()
        self.Tbottom_coh = dict()
        self.full_weight = dict()

        for l in range(nlayer):
            eps_lm1 = permittivity[l - 1] if l > 0 else 1
            eps_l = permittivity[l]
            if l < nlayer - 1:
                eps_lp1 = permittivity[l + 1]

            nsl = streams.n[l]  # number of streams in layer l
            nslm1 = streams.n[l - 1] if l > 0 else streams.n_air  # number of streams * npol in the layer l - 1 (lm1)
            nslp1 = streams.n[l + 1] if l < nlayer - 1 else streams.n_substrate  # number of streams * npol in the layer l + 1 (lp1)

            # compute reflection coefficient between l and l - 1  UP
            # snow-snow UP
            self.Rtop_coh[l] = interfaces[l].specular_reflection_matrix(frequency, eps_l, eps_lm1,
                                                                        streams.mu[l], npol)

            # compute transmission coefficient between l and l - 1 UP
            ns_common_top = min(nsl, nslm1)
            # snow-snow or air UP
            self.Ttop_coh[l] = interfaces[l].coherent_transmission_matrix(frequency, eps_l, eps_lm1,
                                                                          streams.mu[l][0:ns_common_top], npol)

            # compute transmission coefficient between l and l + 1  DOWN
            if l < nlayer - 1:
                ns_common_bottom = min(nsl, nslp1)
                # snow-snow DOWN
                self.Tbottom_coh[l] = interfaces[l].coherent_transmission_matrix(frequency, eps_l, eps_lp1,
                                                                                 streams.mu[l][0:ns_common_bottom], npol)
            elif substrate is not None:
                # sub-snow
                self.Tbottom_coh[nlayer - 1] = substrate.emissivity_matrix(frequency, eps_l, streams.mu[l], npol)

            # compute reflection coefficient between l and l + 1  DOWN
            if l < nlayer - 1:
                # snow-snow DOWN
                self.Rbottom_coh[l] = interfaces[l].specular_reflection_matrix(frequency, eps_l, eps_lp1, streams.mu[l], npol)
                self.Rbottom_diff[l] = 0
            elif substrate is not None:
                # snow-substrate
                self.Rbottom_coh[l] = substrate.specular_reflection_matrix(frequency, eps_l, streams.mu[l], npol)
                if hasattr(substrate, "ft_even_diffuse_reflection_matrix"):
                    # snow-sub
                    self.Rbottom_diff[l] = substrate.ft_even_diffuse_reflection_matrix(frequency, eps_l,
                                                                                       streams.mu[l], streams.mu[l], m_max, npol)
                else:
                    self.Rbottom_diff[l] = smrt_matrix(0)
            else:
                self.Rbottom_coh[l] = smrt_matrix(0)  # fully absorbant substrate
                self.Rbottom_diff[l] = 0

        # air-snow DOWN
        self.Tbottom_coh[-1] = interfaces[0].coherent_transmission_matrix(frequency, 1, permittivity[0], streams.outmu, npol)
        # air-snow DOWN
        self.Rbottom_coh[-1] = interfaces[0].specular_reflection_matrix(frequency, 1, permittivity[0], streams.outmu, npol)
        self.Rbottom_diff[-1] = 0

    def reflection_top(self, l, m, compute_coherent_only):
        return self.Rtop_coh[l].compress(mode=m, auto_reduce_npol=True)

    def reflection_bottom(self, l, m, compute_coherent_only):

        R = self.Rbottom_coh[l].compress(mode=m, auto_reduce_npol=True)

        if not compute_coherent_only and self.Rbottom_diff[l] is not 0 and not self.Rbottom_diff[l].isnull():
            if m == 0:
                coef = 2 * np.pi
                npol = 2
            else:
                coef = np.pi
                npol = 3

            full_weight = coef * np.repeat(self.streams.weight[l], npol)

            Rdiff = self.Rbottom_diff[l].compress(mode=m, auto_reduce_npol=True)

            R += Rdiff @ smrt_diag(full_weight)

        return R

    def transmission_top(self, l, m, compute_coherent_only):
        return self.Ttop_coh[l].compress(mode=m, auto_reduce_npol=True)

    def transmission_bottom(self, l, m, compute_coherent_only):
        return self.Tbottom_coh[l].compress(mode=m, auto_reduce_npol=True)


class Streams(object):
    __slot__ = 'n', 'mu', 'weight', 'outmu', 'outweight', 'n_substrate', 'n_air'


def compute_stream(n_max_stream, permittivity, permittivity_substrate, mode="most_refringent"):
    #     """Compute the optimal angles of each layer. Use for this a Gauss-Legendre quadrature for the most refringent layer and
    # use Snell-law to prograpate the direction in the other layers takig care of the total reflection.

    #     :param n_max_stream: number of stream
    #     :param permittivity: permittivity of each layer
    #     :type permittivity: ndarray
    #     :returns: mu, weight, outmu
    # """

    streams = Streams()

    #  ### search and proceed with the most refringent layer
    k_most_refringent = np.argmax(permittivity)
    real_index_air = np.real(np.sqrt(permittivity[k_most_refringent] / 1.0))

    if mode is None or mode == "most_refringent":
        # calculate the gaussian weights and nodes for the most refringent layer
        mu_most_refringent, weight_most_refringent = gaussquad(n_max_stream)

    elif mode == "air":

        raise Warning("This code has not been tested yet. Use with caution.")

        def number_stream_in_air(n_stream_densest_layer):
            mu_most_refringent, weight_most_refringent = gaussquad(int(n_stream_densest_layer))
            relsin = real_index_air * np.sqrt(1 - mu_most_refringent ** 2)
            return np.sum(relsin < 1) - n_max_stream

        streams.n = scipy.optimize.brentq(n_max_stream, 2 * n_max_stream)
        mu_most_refringent, weight_most_refringent = gaussquad(streams.n)

    else:
        raise RuntimeError("Unknow mode to compute the number of stream")

    nlayer = len(permittivity)

    #  calculate the nodes and weights of all the other layers

    # calculate real part of the index. It is an approximation.
    # See for instance "Semiconductor Physics, Quantum Electronics & Optoelectronics. 2001. V. 4, N 3. P. 214-218."
    real_index = np.real(np.sqrt(permittivity[k_most_refringent] / permittivity[:]))

    # calculate the angles (=node)
    relsin = real_index[:, np.newaxis] * np.sqrt(1 - mu_most_refringent[np.newaxis, :]**2)

    real_reflection = relsin < 1  # mask where real reflection occurs

    mu = np.zeros((nlayer, n_max_stream), dtype=np.float64)
    mu[real_reflection] = np.sqrt(1 - relsin[real_reflection]**2)

    # calculate the number of streams per layer
    streams.mu = [mu[l, real_reflection[l, :]] for l in range(nlayer)]
    streams.n = np.sum(real_reflection, axis=1)

    assert(all(streams.n > 2))

    # calculate the weight ("a" in Y-Q Jin)
    # weight(1,k)=1-0.5 * (mu(1,k)+mu(2,k))
    # weight(nsk,k)=0.5 * (mu(nsk-1,k)+mu(nsk,k))
    # weight(2:nsk-1,k)=0.5 * (mu(1:nsk-2,k)-mu(3:nsk,k))

    streams.weight = [np.empty_like(mu) for mu in streams.mu]
    for l in range(nlayer):   # TODO: parallel
        streams.weight[l][0] = 1 - 0.5 * (streams.mu[l][0] + streams.mu[l][1])
        streams.weight[l][-1] = np.abs(0.5 * (streams.mu[l][-2] + streams.mu[l][-1]))
        streams.weight[l][1:-1] = np.abs(0.5 * (streams.mu[l][0:-2] - streams.mu[l][2:]))

    # ### calculate the angles (=node) in the air
    # real_index = np.real(np.sqrt(permittivity[0]/1.0))
    # relsin = real_index * np.sqrt(1 - mu[0, :]**2)

    # real_reflection = relsin < 1
    # outmu = np.sqrt(1 - relsin[real_reflection]**2)

    relsin = real_index_air * np.sqrt(1 - mu_most_refringent[:]**2)

    real_reflection = relsin < 1
    streams.outmu = np.sqrt(1 - relsin[real_reflection]**2)
    streams.n_air = len(streams.outmu)

    streams.outweight = np.empty_like(streams.outmu)
    streams.outweight[0] = 1 - 0.5 * (streams.outmu[0] + streams.outmu[1])
    streams.outweight[-1] = 0.5 * (streams.outmu[-2] + streams.outmu[-1])
    streams.outweight[1:-1] = 0.5 * (streams.outmu[0:-2] - streams.outmu[2:])

    # compute the number of stream in the substrate
    if permittivity_substrate is None:
        streams.n_substrate = streams.n[-1]  # same as last layer
    else:
        real_index = np.real(np.sqrt(permittivity_substrate / permittivity[-1]))

        # calculate the angles (=node) in the substrate
        relsin = real_index * np.sqrt(1 - mu_most_refringent[:]**2)
        streams.n_substrate = np.sum(relsin < 1)   # count where real reflection occurs

    return streams


def gaussquad(n):
    #     """return the gauss-legendre roots and weight, only the positive roots are return.

    #     :param n: number of (positive) points in the quadrature. Must be larger than 2
    # """
    assert n >= 2

    mu, weight = scipy.special.orthogonal.p_roots(2 * n)

    mu = mu[-1:n - 1:-1]
    weight = weight[-1:n - 1:-1]

    return mu, weight
