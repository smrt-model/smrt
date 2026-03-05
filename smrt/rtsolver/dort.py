# coding: utf-8

"""
Provide the Discrete Ordinate and Eigenvalue Solver as a multi-stream solver of the radiative transfer model in active
and passive mode.

This solver is precise but less efficient than 2 or 6 flux solvers. Different flavours of DORT (or DISORT) exist in the
literature depending on the mode (passive or active), the polarization capabilities, the density of the medium (sparse
media have trivial inter-layer boundary conditions), on the way the streams are connected between the layers and on the
way the phase function is prescribed. The actual version is a blend between Picard et al. 2004 (active mode for sparse
media) and DMRT-ML (Picard et al. 2013) which works in passive mode only for snow. The DISORT often used in optics
(Stamnes et al. 1988) only works for sparse medium and uses a development of the phase function in Legendre polynomials
on theta. The version used in DMRT-QMS (L. Tsang's group) is similar to the present implementation except it uses spline
interpolation to connect constant-angle streams between the layers although we use direct connection by varying the
angle according to Snell's law. A practical consequence is that the number of streams vary (due to internal reflection)
and the value `n_max_stream` only applies in the most refringent layer. The number of outgoing streams in the air is
usually smaller, sometimes twice smaller (depends on the density profile). It is important not to set too low a value
for `n_max_stream`. E.g. 32 is usually fine, 64 or 128 are better but simulations will be much slower.

Note:
    The DORT solver is very robust in passive mode but may raise exception in active mode due to a matrix
    diagonalisation problem. The exception provides detailed information on how to address this issue. Two new
    diagonalisation approaches were added in January 2024. They are activated by setting the `diagonalization_method`
    optional argument (see :py:mod:`smrt.core.make_model`). The first method (``diagonalization_method='shur'``)
    replaces the scipy.linalg.eig function by a shur decomposition followed by a diagonalisation of the shur matrix.
    While scipy.linalg.eig performs such a shur decomposition internally in any case, it seems that explicitly calling
    the shur decomposition beforehand improves the stability. Nevertheless to really solve the problem, the second
    method (``diagonalization_method='shur_forcedtriu'``) consists in removing the 2x2 and 3x3 blocks from the shur
    matrix, i.e. forcing the shur matrix to be upper triangular (triu in numpy jargon = zeroing the lower part of this
    matrix). This problem is due to the structure of the matrix to be diagonalized and the formulation of the DORT
    method in the polarimetric configuration. The eigenvalues come by triplets and can be very close to each other for
    the three H, V, U Stokes components when scattering is becoming small (or equiv. the azimuth mode 'm' is large). As
    a consequence of the Gershgorin theorem, this results in slightly complex eigenvalues (i.e. eigenvalues with very
    small imaginary part) that comes from 2x2 or 3x3 blocks in the shur decomposition. This would not be a problem if
    the eigenvectors were correctly estimated, but this is not the case. It is indeed difficult to find the correct
    orientation of eigenvectors associated to very close eigenvalues. To overcome the problem, the solution is to remove
    the 2x2 and 3x3 blocks. In principle, it would be safer to check that these blocks are nearly diagonal but this is
    not done in the current implementation. The user is responsible to switch between the options until it works. After
    sufficient successful reports by users are received the last method (forcedtriu) will certainly be the default.

Usage:
    Basic usage with default settings and iba emmodel:
        >>> m = make_model("iba", "dort")

    Handle errors gracefully in batch processing:
        >>> m = make_model("iba", "dort", rtsolver_options = {'error_handling':'nan'})

References:
    - Picard, G., Le Toan, T., Quegan, S., Caraglio, Y., and Castel, T. (2004). Radiative Transfer Modeling of
      Cross-Polarized  Backscatter From a Pine Forest Using the Discrete  Ordinate and Eigenvalue Method. IEEE
      TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 42, NO. 8, https://doi.org/10.1109/TGRS.2004.831229

    - Picard, G., Brucker, L., Roy, A., Dupont, F., Fily, M., Royer, A., and Harlow, C. (2013) Simulation of the
      microwave emission of multi-layered snowpacks using the Dense Media Radiative transfer theory: the DMRT-ML model,
      Geosci. Model Dev., 6, 1061-1078, https://doi.org/10.5194/gmd-6-1061-2013

    - Stamnes, K., Tsay, S-C., Wiscombe, W., and Jayaweera, K. (1988). Numerically stable algorithm for
      discrete-ordinate-method radiative transfer in multiple scattering and emitting layered media. Applied Optics,
      27-12, pp.2502-2509. https://doi.org/10.1364/AO.27.002502
"""

from functools import partial

import joblib

# Stdlib import
# other import
import numpy as np
import scipy.linalg

# local import
from smrt.core.error import SMRTError, smrt_warn
from smrt.core.lib import is_equal_zero, is_zero_scalar, smrt_diag, smrt_matrix
from smrt.core.optional_numba import numba
from smrt.rtsolver.rtsolver_utils import (
    CoherentLayerMixin,
    DiscreteOrdinatesMixin,
    PlanckMixin,
    RTSolverBase,
)


class DORT(RTSolverBase, CoherentLayerMixin, DiscreteOrdinatesMixin, PlanckMixin):
    """
    Implement the Discrete Ordinate and Eigenvalue Solver.

    Args:
        n_max_stream: number of stream in the most refringent layer.
        m_max: number of mode (azimuth).
        stream_mode: If set to "most_refringent" (the default) or "air", streams are
            calculated using the Gauss-Legendre polynomials and then use Snell-law to prograpate the direction in the
            other layers. If set to "uniform_air", streams are calculated uniformly in air and then according to Snells
            law.
        phase_normalization: the integral of the phase matrix should in principe be equal to the scattering coefficient.
            However, some emmodels do not respect this strictly. In general a small difference is due to numerical
            rounding and is acceptable, but a large difference rather indicates either a bug in the emmodel or input
            parameters that breaks the assumption of the emmodel. The most typical case is when the grain size is too
            big compared to wavelength for emmodels that rely on Rayleigh assumption. If this argument is to True, the
            phase matrix is normalized to be coherent with the scattering coefficient, but only when the difference is
            moderate (0.7 to 1.3). If set to "forced" the normalization is always performed. This option is dangerous
            because it may hide bugs or unappropriate input parameters (typically too big grains). If set to False, no
            normalization is performed. If set to "auto" the normalization is performed except for emmodels not
            respecting the reciprocity princple (which the normalization relies on).
        phase_symmetrization: enforce phase function symmetry by replacing the phase function P by (P + P.T)/2
            (simplified).
        error_handling: If set to "exception" (the default), raise an exception in case of error, stopping
            the code. If set to "nan", return a nan, so the calculation can continue, but the result is of course
            unusuable and the error message is not accessible. This is only recommended for long simulations that
            sometimes produce an error.
        process_coherent_layers: Adapt the layers thiner than the wavelegnth using the MEMLS method. The radiative
            transfer theory is inadequate layers thiner than the wavelength and using DORT with thin layers is generally
            not recommended. In some parcticular cases (such as ice lenses) where the thin layer is isolated between
            large layers, it is possible to replace the thin layer by an equivalent reflective interface. This neglects
            scattering in the thin layer, which is acceptable in most case, because the layer is thin. To use this
            option and more generally to investigate ice lenses, it is recommended to read MEMLS documentation on this
            topic.
        prune_deep_snowpack: this value is the optical depth from which the layers are discarded in the calculation.
            It is to be use to accelerate the calculations for deep snowpacks or at high frequencies when the
            contribution of the lowest layers is neglegible. The optical depth is a good criteria to determine this
            limit. A value of about 6 is recommended. Use with care, especially values lower than 6.
        diagonalization_method: This value set the method for the diagonalization in the eigenvalue solver. The defaut
            is "eig", it uses the scipy.linalg.eig function. The "shur" replaces the scipy.linalg.eig function by a shur
            decomposition followed by a diagonalisation of the shur matrix. The "shur_forcedtriu" forces the shur matrix
            to be upper triangular. The "half_rank_eig" is the fastest method but requires symmetry and energy
            conservation which may fail with some EMModels and for some parameters. The "stamnes88" is another half rank
            fast method.
        diagonalization_cache: If "simple", cache the results of the diagonalization to avoid redundant computation.
            This can speed up significantly the computation when many layers have exactly the same scattering properties
            in a snowpack or across snowpacks of a sensitivity analysis where only one or few layers are changed at a
            time. The drawback is that it uses more memory as the simple cache is never emptied. LRU cache could be
            implemented in the future to limit memory usage while style keeping some efficiency. This feature is
            experimental, please report success and failure.
        rayleigh_jeans_approximation: In passive mode, if True, use the Rayleigh-Jeans approximation for the Planck function.
            This mode was used by default up to SMRT 1.5.1, but is not as precise as the full Planck function at higher
            frequencies and low temperatures.
    """

    # this specifies which dimension this solver is able to deal with. Those not in this list must be managed by the
    # caller (Model object) e.g. here, frequency, time, ... are not managed
    _broadcast_capability = {
        "theta_inc",
        "polarization_inc",
        "theta",
        "phi",
        "polarization",
    }

    def __init__(
        self,
        n_max_stream=32,
        m_max=2,
        stream_mode="most_refringent",
        phase_normalization="auto",
        phase_symmetrization=False,
        error_handling="exception",
        process_coherent_layers=False,
        prune_deep_snowpack=None,
        diagonalization_method="eig",
        diagonalization_cache=False,
        rayleigh_jeans_approximation=False,
    ):
        DiscreteOrdinatesMixin.init(self, n_max_stream=n_max_stream, stream_mode=stream_mode, m_max=m_max)
        CoherentLayerMixin.init(self, process_coherent_layers=process_coherent_layers)
        PlanckMixin.init(self, rayleigh_jeans_approximation=rayleigh_jeans_approximation)

        self.phase_normalization = phase_normalization
        self.phase_symmetrization = phase_symmetrization
        self.error_handling = error_handling
        self.diagonalization_method = diagonalization_method
        self.diagonalization_cache = diagonalization_cache
        self.rayleigh_jeans_approximation = rayleigh_jeans_approximation

        if self.phase_symmetrization:
            smrt_warn("symmetrization is under development and it is not sure it is working yet.")

        if prune_deep_snowpack is True:
            prune_deep_snowpack = 6
        self.prune_deep_snowpack = prune_deep_snowpack

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
            snowpack: Snowpack object.
            emmodels: List of electromagnetic models object.
            sensor: Sensor object.
            atmosphere: [Optional] Atmosphere object.

        Returns:
            result: Result object.
        """

        self.init_solve(snowpack, emmodels, sensor, atmosphere)

        self.process_coherent_layers()  # must be before prepare_streams

        self.prepare_streams()

        self.temperature = [layer.temperature for layer in self.snowpack.layers] if self.sensor.mode == "P" else None

        m_max = self.m_max if self.sensor.mode == "A" else 0

        # solve the RT equation
        outmu, intensity = self.dort(m_max=m_max)

        # interpolate to the requested streams
        intensity = self.interpolate_intensity(outmu, intensity)

        return self.make_result(outmu, intensity)

    def dort(self, m_max=0, special_return=False):
        """Solve the radiative transfer equation using the DORT method.

        This is a low-level implementation of the discrete-ordinate and eigenvalue solver.
        It is not intended to be called directly by end users; use :meth:`solve` instead.

        Args:
            m_max (int): Maximum azimuthal mode to compute (0 for passive mode).
            special_return (bool or str): If set to a truthy value or to specific debug flags
                (for example ``'bBC'``), the method may return internal debug structures
                instead of the usual output.

        Returns:
            tuple: ``(outmu, intensity_up)`` where ``outmu`` is the array of outgoing cosines
            and ``intensity_up`` contains the upwelling intensities (shape depends on sensor mode).
        """
        # """

        if self.sensor.mode == "P":
            npol = 2
        elif self.sensor.mode == "A":
            npol = 3
        else:
            raise NotImplementedError()

        # prepare the atmosphere

        self.atmosphere_result = (
            self.atmosphere.run(self.sensor.frequency, self.streams.outmu, npol)
            if self.atmosphere is not None
            else None
        )

        #
        # compute the incident intensity array depending on the sensor

        intensity_0, intensity_higher, incident_streams = (
            self.prepare_incident_intensity()
        )  # TODO Ghi: make an iterator

        #
        # compute interface reflection and transmittance properties

        interfaces = InterfaceProperties(
            self.sensor.frequency,
            self.snowpack.interfaces,
            self.snowpack.substrate,
            self.effective_permittivity,
            self.streams,
            m_max,
            npol,
        )
        #
        # create eigenvalue solvers
        eigenvalue_solver = [
            EigenValueSolver(
                ke=self.emmodels[l].ke,
                ks=self.emmodels[l].ks,
                ft_even_phase_function=self.emmodels[l].ft_even_phase,
                mu=self.streams.mu[l],
                weight=self.streams.weight[l],
                m_max=m_max,
                method=self.diagonalization_method,
                normalization=self.phase_normalization
                if self.phase_normalization != "auto"
                else getattr(self.emmodels[l], "_respect_reciprocity_principle", True),
                symmetrization=self.phase_symmetrization,
                cache=self.diagonalization_cache,
            )
            for l in range(len(self.emmodels))
        ]

        #
        # compute the outgoing intensity for each mode
        #
        if self.sensor.mode == "P":
            intensity_up = np.zeros((npol, self.streams.n_air))
        elif self.sensor.mode == "A":
            intensity_up = np.zeros((npol, self.streams.n_air, npol, len(incident_streams)))
            # compute the coherent contribution
            coherent_intensity_up_0 = self.dort_modem_banded(
                0,
                self.streams,
                eigenvalue_solver,
                interfaces,
                intensity_0,
                compute_coherent_only=True,
            )
        else:
            raise RuntimeError("unknow sensor mode")

        for m in range(0, m_max + 1):
            intensity_down_m = intensity_0 if m == 0 else intensity_higher

            # compute the upwelling intensity for mode m
            intensity_up_m = self.dort_modem_banded(
                m,
                self.streams,
                eigenvalue_solver,
                interfaces,
                intensity_down_m,
                special_return=special_return,
            )

            if special_return:  # debuging
                return intensity_up_m

            if self.sensor.mode == "A":
                # substrate the coherent contribution
                intensity_up_m[0:2, :, 0:2, :] -= coherent_intensity_up_0 * (1 + float(m > 0))

            self.add_intensity_mode(intensity_up, intensity_up_m, m)

            # TODO: implement a convergence test if we want to avoid long computation
            # when self.m_max is too high for the phase function.

        if self.sensor.mode == "P":
            if self.atmosphere_result is not None:
                intensity_up = (
                    self.planck_function(self.atmosphere_result.tb_up)
                    + self.atmosphere_result.transmittance * intensity_up
                )
            intensity_up = self.inverse_planck_function(intensity_up)  # convert back to brightness temperature
            outmu = self.streams.outmu
        elif self.sensor.mode == "A":
            # compress to get only the backscatter
            backscatter_intensity_up = np.empty((npol, npol, len(incident_streams)))
            for j, i in enumerate(incident_streams):
                # the j-th column vector contains the stream i, with angle mu[i]
                # at the same time, convert the dimension to pola, pola, incidence
                backscatter_intensity_up[:, :, j] = intensity_up[:, i, :, j]

            outmu = self.streams.outmu[incident_streams]
            intensity_up = backscatter_intensity_up
        else:
            raise NotImplementedError()

        return outmu, intensity_up

    def prepare_incident_intensity(self):
        if self.sensor.mode == "A":
            # send a direct beam

            # incident angle at a given angle
            # use interpolation to get the based effective angle

            #
            # delta(x) = 1/2pi + 1/pi*sum_{n=1}{infinty} cos(nx)
            #

            incident_streams = self.prepare_incident_streams()

            intensity_0 = np.zeros((2 * self.streams.n_air, 2 * len(incident_streams)))
            intensity_higher = np.zeros((3 * self.streams.n_air, 3 * len(incident_streams)))
            # 2 and 3 are for the polarizations

            j0 = 0
            j_higher = 0
            for i in incident_streams:
                power = 1.0 / (2 * np.pi * self.streams.outweight[i])
                for ipol in [0, 1]:
                    intensity_0[2 * i + ipol, j0] = power
                    j0 += 1
                for ipol in [0, 1, 2]:
                    intensity_higher[3 * i + ipol, j_higher] = 2 * power
                    j_higher += 1

        elif self.sensor.mode == "P":
            npol = 2
            incident_streams = []

            if self.atmosphere_result is not None:
                # incident radiation is a function of frequency and incidence angle
                # assume azimuthally symmetric
                intensity_0 = self.planck_function(self.atmosphere_result.tb_down)
                assert intensity_0.shape == (npol, self.streams.n_air)
                # convert pola, theta to (theta, pola) and add batch dimension
                intensity_0 = np.swapaxes(intensity_0, 0, 1).flatten()[:, np.newaxis]
                intensity_higher = np.zeros_like(intensity_0)
            else:
                intensity_0 = np.zeros((npol * self.streams.n_air, 1))
                intensity_higher = intensity_0
                intensity_0.flags.writeable = False  # make immutable
                intensity_higher.flags.writeable = False  # make immutable
        else:
            raise SMRTError("Unknow sensor mode")

        return intensity_0, intensity_higher, incident_streams

    def dort_modem_banded(
        self,
        m,
        streams,
        eigenvalue_solver,
        interfaces,
        intensity_down_m,
        compute_coherent_only=False,
        special_return=False,
    ):
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
            nband = npol * max(
                np.max(2 * streams.n[1:] + streams.n[:-1]),
                np.max(streams.n[1:] + 2 * streams.n[:-1]),
            )
            # print("gain:", nband / (3 * npol * np.max(streams.)))
            # in principle could be better optimized as the number of upper and lower diagonal can be different
        else:
            nband = 3 * npol * np.max(streams.n)  # each layer appears in 3 blocks
        # (bottom, top of the current layer, and top of layer below (for downward directons) and
        # bottom of the layer above (for upward directions)

        # Boundary condition matrix
        bBC = np.zeros((2 * nband + 1, nboundary))  # we use banded Boundary condition matrix

        # rhs vector size
        assert len(intensity_down_m.shape) == 2
        nvector = intensity_down_m.shape[1]
        b = np.zeros((nboundary, nvector))

        nlayer = len(eigenvalue_solver)

        # used to estimate if the medium is deep enough
        optical_depth = 0

        for l in range(0, nlayer):
            nsl = streams.n[l]  # number of streams in layer l
            nsl_npol = nsl * npol  # number of streams * npol in layer l
            nsl2_npol = 2 * nsl_npol  # number of eigenvalues in layer l (should be 2 * npol*n_stream)
            nslm1_npol = (
                (streams.n[l - 1] * npol) if l > 0 else (streams.n_air * npol)
            )  # number of streams * npol in the layer l - 1 (lm1)
            # number of streams * npol in the layer l + 1 (lp1)
            nslp1_npol = (streams.n[l + 1] * npol) if l < nlayer - 1 else None

            # solve the eigenvalue problem for layer l

            # TODO: the following duplicates the eignevalue_solver call line. A better way should be implemented,
            # either with a variable holding the exception type (
            # and use of a never raised exception or see contextlib module if useful)
            if self.error_handling == "nan":
                try:
                    # run in a try to catch the exception
                    beta, Eu, Ed = eigenvalue_solver[l].solve(m, compute_coherent_only)
                except SMRTError:
                    return _reshape_output(np.full_like(intensity_down_m, np.nan).squeeze(), npol)
            else:
                beta, Eu, Ed = eigenvalue_solver[l].solve(m, compute_coherent_only)
            assert Eu.shape[0] == npol * nsl

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
            _todiag(bBC, il_topl, j, _matmul(Ed - _matmul(Rtop_l, Eu), transt))

            if l < nlayer - 1:
                Tbottom_lp1 = interfaces.transmission_bottom(l, m, compute_coherent_only)
                # the size of Tbottom_lp1 can be the nsl_npol in general or nslp1_npol if only the specular is present
                # and some streams are subject to total reflection.
                if not is_equal_zero(Tbottom_lp1):
                    ns_npol_common_bottom = min(Tbottom_lp1.shape[0], nslp1_npol)
                    _todiag(bBC, il_top[l + 1], j, -_matmul(Tbottom_lp1, Ed, transb)[:ns_npol_common_bottom, :])

            # fill the vector
            if m == 0 and self.temperature is not None and self.temperature[l] > 0:
                if is_equal_zero(Rtop_l):
                    b[il_topl : il_topl + nsl_npol, :] -= self.planck_function(
                        self.temperature[l]
                    )  # to be put at layer (l)
                else:
                    b[il_topl : il_topl + nsl_npol, :] -= (
                        (1.0 - _muleye(Rtop_l)) * self.planck_function(self.temperature[l])
                    )[:, np.newaxis]  # a mettre en (l)
                # the muleye comes from the isotropic emission of the black body

                if l < nlayer - 1 and self.temperature[l] > 0 and not is_equal_zero(Tbottom_lp1):
                    b[il_top[l + 1] : il_top[l + 1] + ns_npol_common_bottom, :] += (
                        _muleye(Tbottom_lp1) * self.planck_function(self.temperature[l])
                    )[:ns_npol_common_bottom, np.newaxis]  # to be put at layer (l + 1)

            if l == 0:  # Air-snow interface
                Tbottom_air_down = interfaces.transmission_bottom(-1, m, compute_coherent_only)
                if not is_equal_zero(Tbottom_air_down):
                    ns_npol_common_bottom = min(Tbottom_air_down.shape[0], nsl_npol)  # see the comment on Tbottom_lp1
                    b[il_topl : il_topl + ns_npol_common_bottom, :] += _matmul(Tbottom_air_down, intensity_down_m)

            # -------------------------------------------------------------------------------
            # Eq 18 & 22 BOTTOM of layer l

            # compute reflection coefficient between l and l + 1
            Rbottom_l = interfaces.reflection_bottom(l, m, compute_coherent_only)

            # fill the matrix
            _todiag(bBC, il_bottoml, j, _matmul(Eu - _matmul(Rbottom_l, Ed), transb))

            if l > 0:
                Ttop_lm1 = interfaces.transmission_top(l, m, compute_coherent_only)
                if not is_equal_zero(Ttop_lm1):
                    ns_npol_common_top = min(Ttop_lm1.shape[0], nslm1_npol)  # see the comment on Tbottom_lp1
                    _todiag(
                        bBC, il_bottom[l - 1], j, -_matmul(Ttop_lm1, Eu, transt)[:ns_npol_common_top, :]
                    )  # to be put at layer (l - 1)

            # fill the vector
            if m == 0 and self.temperature is not None and self.temperature[l] > 0:
                if is_equal_zero(Rbottom_l):
                    b[il_bottoml : il_bottoml + nsl_npol, :] -= self.planck_function(
                        self.temperature[l]
                    )  # to be put at layer (l)
                else:
                    b[il_bottoml : il_bottoml + nsl_npol, :] -= (
                        (1.0 - _muleye(Rbottom_l)) * self.planck_function(self.temperature[l])
                    )[:, np.newaxis]  # to be put at layer (l)
                if l > 0 and not is_equal_zero(Ttop_lm1):
                    b[il_bottom[l - 1] : il_bottom[l - 1] + ns_npol_common_top, :] += (
                        _muleye(Ttop_lm1) * self.planck_function(self.temperature[l])
                    )[:ns_npol_common_top, np.newaxis]  # to be put at layer (l - 1)

            if (
                m == 0
                and l == nlayer - 1
                and self.snowpack.substrate is not None
                and self.snowpack.substrate.temperature is not None
                and self.temperature is not None
            ):
                Tbottom_sub = interfaces.transmission_bottom(l, m, compute_coherent_only)
                ns_npol_common_bottom = min(Tbottom_sub.shape[0], nsl_npol)  # see the comment on Tbottom_lp1
                if not is_equal_zero(Tbottom_sub):
                    b[il_bottoml : il_bottoml + ns_npol_common_bottom, :] += (
                        _muleye(Tbottom_sub) * self.planck_function(self.snowpack.substrate.temperature)
                    )[:ns_npol_common_bottom, np.newaxis]  # to be put at layer  (l)

            # Finalize
            optical_depth += np.min(np.abs(beta)) * self.snowpack.layers[l].thickness

            if self.prune_deep_snowpack is not None and optical_depth > self.prune_deep_snowpack:
                # prune the matrix and vector
                nboundary = sum(streams.n[0 : l + 1]) * 2 * npol
                bBC = bBC[:, 0:nboundary]
                b = b[0:nboundary, :]

                break

        # -------------------------------------------------------------------------------
        #   solve the boundary system BCx=b

        if special_return == "bBC":
            return bBC, b

        if self.snowpack.substrate is None and optical_depth < 5:
            smrt_warn(
                "DORT has detected that the snowpack is optically shallow (tau=%g) and no substrate has been set, meaning that the space "
                "under the snowpack is 'empty' with snowpack shallow enough to affect the measured signal at the surface."
                "This is usually not wanted and can produce wrong results. Either increase the thickness of the snowpack or set a substrate."
                " If wanted, add a transparent substrate to supress this warning" % optical_depth
            )

        x = scipy.linalg.solve_banded((nband, nband), bBC, b, overwrite_ab=True, overwrite_b=True)

        # #  ! calculate the intensity emerging from the snowpack
        l = 0
        j = jl[l]  # should be 0
        nsl_npol = streams.n[l] * npol
        nsl2_npol = 2 * nsl_npol
        I1up_m = Eu_0 @ transt_0 @ x[j : j + nsl2_npol, :]

        if m == 0 and self.temperature is not None and self.temperature[0] > 0:
            I1up_m += self.planck_function(self.temperature[0])  # just under the interface

        Rbottom_air_down = interfaces.reflection_bottom(-1, m, compute_coherent_only)
        Ttop_0 = interfaces.transmission_top(0, m, compute_coherent_only)  # snow-air

        I0up_m = _matmul(Rbottom_air_down, intensity_down_m) + _matmul(Ttop_0, I1up_m)[0 : streams.n_air * npol, :]

        I0up_m = np.array(I0up_m).squeeze()

        return _reshape_output(I0up_m, npol)


def _reshape_output(I0up_m, npol: int):
    """Reshape the intensity output into the expected polarization/angle layout.

    Args:
        I0up_m (ndarray): Input intensity array. Can be 1D or 2D depending on context.
        npol (int): Number of polarization components (typically 2 or 3).

    Returns:
        ndarray: Reshaped intensity array. If input is 1D, returns an array shaped
            `(pola, theta)`. If input is 2D, returns an array shaped
            `(pola, theta, pola_in, theta_in)` and transposed to `(pola, theta, pola_in, theta_in)`.
    """
    if np.ndim(I0up_m) == 1:
        # split the outgoing polarization:
        I0up_m = I0up_m.reshape((I0up_m.shape[0] // npol, npol)).transpose()
    elif np.ndim(I0up_m) == 2:
        # split the incoming and outgoing polarization:
        I0up_m = I0up_m.reshape((I0up_m.shape[0] // npol, npol, I0up_m.shape[1] // npol, npol)).transpose(1, 0, 3, 2)
    else:
        raise RuntimeError("Should not be here. Not implemented np.ndim(I0up_m) > 2.")
    return I0up_m


def _muleye(x):
    """Return the sum/diagonal appropriate to different matrix types.

    Args:
        x: Can be an instance of ``smrt_diag``, a scalar, or a 2D array.

    Returns:
        ndarray: If ``x`` is ``smrt_diag``, returns its diagonal as 1D array. If ``x`` is a scalar,
        returns at least a 1D array with that value. If ``x`` is a 2D array, returns the row-wise sum.
    """

    if isinstance(x, smrt_diag):
        return x.diagonal()
    elif (is_zero_scalar(x)) or (len(x.shape) == 0):
        return np.atleast_1d(x)
    else:
        assert len(x.shape) == 2
        return np.sum(x, axis=1)


def _matmul(a, b, *args):
    """Multiply arrays or scalars, supporting chaining and scalar*array semantics.

    This wrapper multiplies ``a`` and ``b``; if additional arrays are provided in ``*args``,
    they are multiplied in sequence (left-associative). If either operand is scalar, scalar
    multiplication is used.

    Args:
        a: Scalar or array-like left operand.
        b: Scalar or array-like right operand.
        *args: Optional additional operands to multiply in sequence.

    Returns:
        The result of the chained multiplication.
    """

    if args:
        b = _matmul(b, *args)
    if np.isscalar(a) or np.isscalar(b):
        return a * b
    else:
        return a @ b


def _todiag(bmat, oi, oj, dmat):
    """Insert a dense block ``dmat`` into a banded matrix representation ``bmat``.

    The function assumes ``bmat`` uses the (upper, lower) banded storage with center row
    at index ``u = (bmat.shape[0] - 1) // 2`` and inserts the values of ``dmat`` at the
    appropriate diagonals given offsets ``oi`` and ``oj``.

    Args:
        bmat (ndarray): Banded matrix storage array of shape ``(2*nband+1, nboundary)``.
        oi (int): Row offset where the block should be placed.
        oj (int): Column offset where the block should be placed.
        dmat (ndarray): Dense matrix block to insert.
    """

    u = (bmat.shape[0] - 1) // 2

    n, m = dmat.shape
    dmat_flat = dmat.flatten()

    for j in range(0, m):
        ldiag = min(n, m - j)
        bmat[u + oi - oj - j, j + oj : j + oj + ldiag] = dmat_flat[j : j + (m + 1) * ldiag : m + 1]

    for i in range(1, n):
        ldiag = min(n - i, m)
        bmat[u + oi - oj + i, 0 + oj : 0 + oj + ldiag] = dmat_flat[i * m : i * m + (m + 1) * ldiag : m + 1]


if numba is not None:
    compiled_todiag = numba.jit(nopython=True, cache=True)(_todiag)

    def _todiag(bmat, oi, oj, dmat):
        compiled_todiag(bmat, int(oi), int(oj), dmat)


# not used anymore
# def extend_2pol_npol(x, npol):
#     if npol == 2:
#         return x

#     if scipy.sparse.isspmatrix_dia(x):
#         raise Exception("Should not arrive in this branch, contact the developer if it does")
#         y = scipy.sparse.diags(extend_2pol_npol(x.diagonal(), npol))
#     elif len(x.shape) == 1:
#         y = np.zeros(len(x) // 2 * npol)
#         y[0::npol] = x[0::2]
#         y[1::npol] = x[1::2]
#     elif len(x.shape) == 2:
#         y = np.zeros((x.shape[0] // 2 * npol, x.shape[1] // 2 * npol))
#         y[0::npol, 0::npol] = x[0::2, 0::2]
#         y[0::npol, 1::npol] = x[0::2, 1::2]
#         y[1::npol, 0::npol] = x[1::2, 0::2]
#         y[1::npol, 1::npol] = x[1::2, 1::2]
#     else:
#         raise SMRTError("should never be here")

#     return y


_eigenvaluesolver_diagnalization_simple_cache = {}


class EigenValueSolver(object):
    def __init__(self, ke, ks, ft_even_phase_function, mu, weight, m_max, normalization, symmetrization, method, cache):
        """Initialize the EigenValueSolver.

        Args:
            ke (callable): Extinction coefficient of the layer for mode `m`.
            ks (callable or float): Scattering coefficient of the layer for mode `m`.
            ft_even_phase_function (callable or None): Function that returns the even part of the
                phase function. Should accept `(mu_s, mu_i, m_max)` and return a matrix-like object.
            mu (array-like): Cosines of the stream angles (positive values for outgoing).
            weight (array-like): Quadrature weights for the streams.
            m_max (int): Maximum azimuthal mode to consider.
            normalization (str or bool): Phase-function normalization option (e.g. ``'auto'``, ``'forced'``,
                or ``False``).
            symmetrization (bool): If True, enforce phase-function symmetry.
            method (str): Diagonalization method. Supported values: ``'eig'``, ``'shur'``,
                ``'shur_forcedtriu'``, ``'half_rank_eig'``, ``'stamnes88'``.
            cache (bool): If True, cache diagonalization results to avoid recomputation.

        """
        # :param Ke: extinction coefficient of the layer for mode m
        # :param ft_even_phase: ft_even_phase function of the layer for mode m
        # :param mu: cosines
        # :param weight: weights

        self.ke = ke
        self.ks = ks
        self.ft_even_phase_function = ft_even_phase_function
        self.m_max = m_max
        self.mu = mu
        self.weight = weight
        self.normalization = normalization
        self.symmetrization = symmetrization

        # default: use the solve_generic method and compute the full matrix
        # these defaults may be overwritten in the next if/elif
        self.solve = self.solve_generic
        self.compute_half_rank_phase = False

        self.method = method
        self.cache = cache

        match self.method:
            case "eig":
                self.diagonalize_function = self.diagonalize_eig
            case "shur":
                self.diagonalize_function = partial(self.diagonalize_shur, force_triu=False)
            case "shur_forcedtriu":
                self.diagonalize_function = partial(self.diagonalize_shur, force_triu=True)
            case "half_rank_eig":
                self.compute_half_rank_phase = True
                self.diagonalize_function = self.diagonalize_half_rank_eig
            case "stamnes88":
                self.compute_half_rank_phase = True
                self.solve = self.solve_stamnes88
            case _:
                raise SMRTError(f"Unknown method '{method}' to diagonalize the matrix")

        self.norm_0 = None
        self.norm_m = None

    def compute_ft_even_phase(self):
        # cached version of the ft_even_phase
        if not hasattr(self, "_ft_even_phase"):
            if self.ft_even_phase_function is None:
                self._ft_even_phase = smrt_matrix(0)
            else:
                fullmu = np.concatenate((self.mu, -self.mu))
                if self.compute_half_rank_phase and not self.symmetrization:
                    # compute only mu_s > 0 for all mu_i >0 and <0
                    self._ft_even_phase = self.ft_even_phase_function(self.mu, fullmu, self.m_max)
                else:
                    # compute the full matrix for all mu_s and mu_i >0 and <0
                    self._ft_even_phase = self.ft_even_phase_function(fullmu, fullmu, self.m_max)

        return self._ft_even_phase

    def solve(self, m, compute_coherent_only, debug_A=False):
        raise RuntimeError(
            "This method is the entry point and it must be set to one of the solve function at initialization"
        )

    def solve_generic(self, m, compute_coherent_only, debug_A=False):
        # solve the homogeneous equation for a single layer and return eignen values and eigen vectors
        # :param m: mode
        # :param compute_coherent_only
        # :returns: beta, E, Q
        #

        # this coefficient come from the 1/4pi normalization of the RT equation and the
        # 1/(4*pi) * int_{phi=0}^{2*pi} cos(m phi)*cos(n phi) dphi
        # note that equation A7 and A8 in Picard et al. 2018 has an error, it does not show this coefficient.

        # calculate the A matrix. Eq (12),  or 0 if compute_coherent_only
        if compute_coherent_only:
            return self.return_no_scattering(m)

        A = self.compute_ft_even_phase().compress(mode=m, auto_reduce_npol=True)

        if is_equal_zero(A):
            return self.return_no_scattering(m)

        if self.symmetrization:
            A = symmetrize_phase_matrix(A, m)

        npol = 2 if m == 0 else 3
        # compute invmu
        invmu = 1.0 / self.mu
        invmu = np.repeat(invmu, npol)
        invmu = np.concatenate((invmu, -invmu))  # matrix M^-1 in Stamnes et al. DISORT 1988
        fullmu = np.concatenate((self.mu, -self.mu))

        coef = 0.5 if m == 0 else 0.25

        # the phase function is not null, let's continue to create the A matrix
        coef_weight = np.tile(
            np.repeat(-coef * self.weight, npol), 2
        )  # could be cached (per layer) because same for each mode

        A *= coef_weight[np.newaxis, :]

        k = A.shape[0]  # can be n or 2*n depending on whether the symmetry optimization is used or not

        # normalize
        if self.normalization:
            if callable(self.ks):
                A = self.normalize(m, A, self.ks(fullmu[0 : k // npol], npol=npol).compress().diagonal())
            else:
                A = self.normalize(m, A, self.ks)
        # normalization is done

        A[np.diag_indices(k)] += self.ke(fullmu[0 : k // npol], npol=npol).compress().diagonal()
        A = invmu[0:k, np.newaxis] * A

        if debug_A:  # this is not elegant but can be useful. A dedicated function to compute_A is not better
            return A

        # simple caching mechanism to avoid recomputing the diagonalization of identical matrices
        if self.cache:
            key = joblib.hash(A) + self.method
            if key not in _eigenvaluesolver_diagnalization_simple_cache:
                _eigenvaluesolver_diagnalization_simple_cache[key] = self.diagonalize_function(m, A)
            return _eigenvaluesolver_diagnalization_simple_cache[key]
        else:
            # not cached version
            return self.diagonalize_function(m, A)
        # !-----------------------------------------------------------------------------!

    def return_no_scattering(self, m: int):
        # the solution is trivial
        npol = 2 if m == 0 else 3
        n = npol * len(self.mu)

        invmu = 1.0 / self.mu
        invmu = np.repeat(invmu, npol)
        invmu = np.concatenate((invmu, -invmu))  # matrix M^-1 in Stamnes et al. DISORT 1988
        fullmu = np.concatenate((self.mu, -self.mu))

        beta = invmu * self.ke(fullmu, npol=npol).compress().diagonal()
        E = np.eye(2 * n, 2 * n)
        Eu = E[0:n, :]  # upwelling
        Ed = E[n:, :]  # downwelling

        return beta, Eu, Ed

    def normalize(self, m, A, ks):
        # normalize A to conserve energy, assuming isotrope scattering coefficient
        # ks should be a function of mu if non-isotropic medium and to be consistent with ke which is a function of mu

        if m == 0:
            if np.any(ks == 0):  # can't perform normalization
                return A
            self.norm_0 = -ks / np.sum(A, axis=1)
            norm = self.norm_0

            if self.normalization != "forced" and np.any(np.abs(self.norm_0 - 1.0) > 0.3):
                print("norm=", norm)
                raise SMRTError("""The re-normalization of the phase function exceeds the predefined threshold of 30%.
This is likely because of a too large grain size or a bug in the phase function. It is recommended to check the grain size.
You can also deactivate this check using normalization="forced" as an options of the dort solver. It is at last possible
to disable this error raise and return NaN instead by adding the argument rtsolver_options=dict(error_handling='nan') to make_model).""")
        else:
            if self.norm_m is None:
                if self.norm_0 is None:  # be careful, this code is not re-entrant
                    raise Exception(
                        "For the normalization, it is necessary to call this function for the mode m=0 first."
                    )
                # transform the norm_0 for npol
                npol = 2 if m == 0 else 3
                self.norm_m = np.empty(len(self.norm_0) // 2 * npol)
                self.norm_m[0::npol] = self.norm_0[0::2]
                self.norm_m[1::npol] = self.norm_0[1::2]
                for ipol in range(2, npol):
                    # this approach is empirical
                    self.norm_m[ipol::npol] = np.sqrt(self.norm_0[0::2] * self.norm_0[1::2])
            norm = self.norm_m

        A *= norm[:, np.newaxis]
        return A

    def diagonalize_eig(self, m, A):
        # diagonalise the matrix. Eq (13)
        try:
            beta, E = scipy.linalg.eig(A, overwrite_a=True)
        except scipy.linalg.LinAlgError:
            raise SMRTError("Eigen value decomposition failed.\n" + self.diagonalization_error_message())

        npol = 2 if m == 0 else 3
        n = npol * len(self.mu)
        Eu = E[0:n, :]  # upwelling
        Ed = E[n:, :]  # downwelling

        return self.validate_eigen(beta, Eu, Ed, m)

    def diagonalize_shur(self, m, A, force_triu=False):
        # diagonalise the matrix. Eq (13) using Shur decomposition. This avoids some instabilities with the direct eig
        # function
        # in addition it is possible to remove the 2x2 or 3x3 blocks that occurs when eigenvalues are close
        # forcing the lower triangular part of the shur matrix to zero solves this problem but is radical
        # a better algorithm would first check that the 2x2 nd 3x3 blocks are nearly diagional (values are very small)

        try:
            T, Z = scipy.linalg.schur(A)
        except scipy.linalg.LinAlgError:
            raise SMRTError("Schur decomposition failed.\n" + self.diagonalization_error_message())
        try:
            if force_triu:
                T[np.tril_indices(T.shape[0], k=-1)] = 0

            beta, E = scipy.linalg.eig(T, overwrite_a=True)
        except scipy.linalg.LinAlgError:
            raise SMRTError(
                "Diagonalization of the schur decomposition failed.\n" + self.diagonalization_error_message()
            )

        E = Z @ E

        # if m >= 0: #== 2:

        #     #print(m, T)
        #     iscomplex_beta = not np.allclose(beta.imag, 0, atol=0)
        #     iscomplex_E = not np.allclose(E.imag, 0, atol=1e-6)

        #     if np.any(beta.imag != 0):
        #         print("pas ok")
        #         # print(m, T)
        #         # print(m, T[np.tril_indices(T.shape[0], k=-1)])
        #         # print(m, np.max(np.abs(T[np.tril_indices_from(T, k=-1)])), np.all(T[np.tril_indices_from(T, k=-1)]==0))
        #         # print("beta.imag=", beta.imag)

        #         beta, E = scipy.linalg.eig(T, overwrite_a=False)

        #         iscomplex_beta = not np.allclose(beta.imag, 0, atol=0)
        #         iscomplex_E = not np.allclose(E.imag, 0, atol=1e-6)

        #         print('new beta.img', m, iscomplex_beta, iscomplex_E, np.all(beta.imag==0), np.all(E.imag==0), beta.imag)
        #         print('isnan', np.any(np.isnan(beta)), np.any(np.isnan(E)))
        #     else:
        #         print("ok")
        #     #print(f"{m}, {iscomplex_beta}, {iscomplex_E}")

        npol = 2 if m == 0 else 3
        n = npol * len(self.mu)
        Eu = E[0:n, :]  # upwelling
        Ed = E[n:, :]  # downwelling

        return self.validate_eigen(beta, Eu, Ed, m)

    def diagonalize_half_rank_eig(self, m, A):
        # solve the eigenvalue problem of A using half-rank A
        # the decomposition can be found in Stamnes et al. DISORT 1988 for 2-stockes vectors (mode m=0)
        # for the 4-stockes, it is necessary to take into account that the 3rd and 4th components change
        # sign when the outgoing mu_s is changing sign
        #
        # We define D=diag([1, 1, -1, -1]). Note that D^2=1 and D=D^-1 (projection matrix). We can show that the system
        # to solve is:
        #  |d u+ / dtau| = |    alpha       beta| |u+|
        #  |d u- / dtau| = |-D*beta*D -D*alpha*D| |u-|
        # which can be transform into
        #  |d  u+ / dtau| = |  alpha beta*D| |u+  |
        #  |d Du- / dtau| = |-beta*D -alpha| |D u-|
        # it follows that D must be applied to beta before solving the eigen value problem and u- after.
        #
        # this approach is found in Garcia and Siewert 1989. and in C.E. Siewert 2000 JQSRT,
        # however their derivation is difficult to follow. It is straigtforward here.

        n = A.shape[1] // 2

        # see Eq (8e) in Stamnes et al. DISORT 1988
        alpha_mat = -A[0:n, 0:n]  # the top half of the A matrix is for the UP outgoing radiation
        beta_mat = -A[0:n, n:]

        # if A.shape[0] > n:
        #      # check that A has the expected form  (for m = 0 only). For m > 0, the 3rd component as
        #     alternating sign, as mentioned earlier.
        #     assert np.allclose(-A[0:n, 0:n], A[n:, n:]), f"A{m}={-A[0:4, 0:4]}\n{A[n:n+4, n:n+4]}"
        #     assert np.allclose(-A[0:n, n:], A[n:, 0:n]), f"A{m}={-A[0:4, n:n+4]}\n{A[n:n+4, 0:4]}"

        if m > 0:
            # apply the diagonal matrix D
            beta_mat[:, 2::3] = -beta_mat[:, 2::3]

        # solve for the sum: Ep = Eu + Ed
        half_rank_A = (alpha_mat - beta_mat) @ (alpha_mat + beta_mat)  # eq 8e in Stamnes et al. DISORT 1988

        # TODO: it seems that there is an even better solution here: https://edepot.wur.nl/210943 Appendix A
        # where this diagonalization is split in two steps where only symmetric matrix diagonalization
        # is necessary. Better stability ??
        # see also Stamnes 1988 EIGENVALUE PROBLEM equation 5 and following.
        # Stamnes 1998 EIGENVALUE proposes a faster method than in Nakajima and Tanaka which involves
        # Cholesky decomposition

        # diagonalise the matrix. Eq (13)
        try:
            beta, Ep = scipy.linalg.eig(half_rank_A, overwrite_a=True)
        except scipy.linalg.LinAlgError:
            raise SMRTError("Diagonalization of the halk rank matrix failed.\n" + self.diagonalization_error_message())

        beta = np.sqrt(beta.real)  # eq 8e in Stamnes et al. DISORT 1988

        # compute the difference of the eigenvecteur Em = Eu - Ed using Eq. 8d in Stamnes et al. DISORT 1988
        Em = (alpha_mat + beta_mat) @ (Ep * (1 / beta)[np.newaxis, :])

        # Eu = np.empty((n, 2 * n))
        # Eu[:, 0:n] = 0.5 * (Ep - Em)
        # Eu[:, n:] = 0.5 * (Ep + Em)
        Eu = np.hstack((0.5 * (Ep - Em), 0.5 * (Ep + Em)))

        # Ed = np.empty((n, 2 * n))
        # Ed[:, 0:n] = Eu[:, n:]
        # Ed[:, n:] = Eu[:, 0:n]
        Ed = np.hstack((Eu[:, n:], Eu[:, 0:n]))

        if m > 0:
            # apply the diagonal matrix D
            Ed[2::3, :] = -Ed[2::3, :]  # apply eq 43b in C.E. Siewert 2000 JQSRT

        beta = np.concatenate((beta, -beta))

        return self.validate_eigen(beta, Eu, Ed, m)

    def solve_stamnes88(self, m, compute_coherent_only, debug_A=False):
        # solve the homogeneous equation for a single layer and return eignen values and eigen vectors
        # :param m: mode
        # :param compute_coherent_only
        # :returns: beta, E, Q
        #
        smrt_warn("The stamnes88 solver is not fully validated. Use for debugging only.")

        npol = 2 if m == 0 else 3

        n = npol * len(self.mu)

        # this coefficient come from the 1/4pi normalization of the RT equation and the
        # 1/(4*pi) * int_{phi=0}^{2*pi} cos(m phi)*cos(n phi) dphi
        # note that equation A7 and A8 in Picard et al. 2018 has an error, it does not show this coefficient.
        coef = 0.5 if m == 0 else 0.25

        # compute inv_mu
        inv_mu = np.repeat(1 / self.mu, npol)

        # calculate the A matrix. Eq (12),  or 0 if compute_coherent_only
        if compute_coherent_only:
            return self.return_no_scattering(m)

        phase = self.compute_ft_even_phase().compress(mode=m, auto_reduce_npol=True)

        if is_equal_zero(phase):
            return self.return_no_scattering(m)

        Cp = phase[0:n, 0:n]
        Cm = phase[0:n, n:]

        # if m > 0:
        #     # apply the diagonal matrix D
        #     Cm[:, 2::3] = - Cm[:, 2::3]

        ke = self.ke(self.mu, npol=npol).compress().diagonal()
        weight = np.repeat(self.weight, npol)
        inv_weight_ke = 1 / weight * ke  # could be cached (per layer) because same for each mode

        # Calculate G+ and G- (Eq 5c)
        Gp = -coef * (Cp + Cm)
        Gm = -coef * (Cp - Cm)

        # add inv_weight to the diagonal. Functionl alternative (slover): np.fill_diagonal(X, X.diagonal() + d)
        Gp.ravel()[:: n + 1] += inv_weight_ke
        Gm.ravel()[:: n + 1] += inv_weight_ke

        # check that Gp and Gm are symmetric:
        # if m > 0:
        #     print(Gp)
        #     print(Gp.T)
        #     print("--")
        if m == 0:
            assert np.allclose(Gp, Gp.T), f"Gp is not symmetric (m={m})"
            assert np.allclose(Gm, Gm.T), f"Gm is not symmetric (m={m})"

        # Calculate X+ and X-

        wm = np.sqrt(weight * inv_mu)

        Xp = wm[:, np.newaxis] * Gp * wm[np.newaxis, :]
        Xm = wm[:, np.newaxis] * Gm * wm[np.newaxis, :]

        # Xm is positive definite.

        L = np.linalg.cholesky(Xm)
        # the lower instead of the upper triangular matrix is return. Switch all .T w/r to Stamnes et al. 1988

        # Calculate the S matrix
        S = L.T @ Xp @ L

        # TODO: to implement the fast approach taking benefit of the symmetry using numba: see Stamnes et aL. 1988

        # diagonalize this positive definite symmetric matrix if m == 0.
        if m == 0:
            beta, V = np.linalg.eigh(S)
        else:
            # S is not symmetric unfrotunately
            try:
                beta, V = scipy.linalg.eig(S, overwrite_a=True)
            except scipy.linalg.LinAlgError:
                raise SMRTError(
                    "Diagonalization of the halk rank matrix failed.\n" + self.diagonalization_error_message()
                )

        beta = np.sqrt(beta)

        Em = scipy.linalg.solve_triangular(L, V, trans="T")

        # we deduce Ep
        Ep = Xm @ (Em * (-1 / beta)[np.newaxis, :])

        Eu = np.hstack((0.5 * (Ep - Em), 0.5 * (Ep + Em)))
        Ed = np.hstack((Eu[:, n:], Eu[:, 0:n]))

        if m > 0:
            # apply the diagonal matrix D
            Ed[2::3, :] = -Ed[2::3, :]  # apply eq 43b in C.E. Siewert 2000 JQSRT

        beta = np.concatenate((beta, -beta))

        return self.validate_eigen(beta, Eu, Ed, m)

    def validate_eigen(self, beta, Eu, Ed, m):
        iscomplex_beta = not np.allclose(beta.imag, 0, atol=np.max(beta.real) * 1e-07)
        iscomplex_Eu = not np.allclose(Eu.imag, 0, atol=1e-6)
        iscomplex_Ed = not np.allclose(Ed.imag, 0, atol=1e-6)
        diagonalization_failed = iscomplex_beta or iscomplex_Eu or iscomplex_Ed

        reasons = []
        if iscomplex_beta:
            reasons.append("Some eigen values beta are complex.")
        if iscomplex_Eu or iscomplex_Ed:
            reasons.append("Some eigen vectors are complex.")

        if diagonalization_failed:
            print("Info: ks:", self.ks, " m:", m)
            mask = np.abs(Eu.imag) > 1e-8
            print("Eu:", Eu[mask], "beta:", beta[np.any(mask, axis=0)])

            raise SMRTError("n".join(reasons) + "\n" + self.diagonalization_error_message())

        if np.iscomplexobj(Eu):
            mask = abs(Eu.imag) > np.linalg.norm(Eu) * 1e-5
            if np.any(mask):
                print(np.any(mask, axis=1))
                print(beta[np.any(mask, axis=1)])
                print(beta)
            else:
                Eu = Eu.real
        if np.iscomplexobj(Ed):
            mask = abs(Ed.imag) > np.linalg.norm(Ed) * 1e-5
            if np.any(mask):
                print(np.any(mask, axis=1))
                print(beta[np.any(mask, axis=1)])
                print(beta)
            else:
                Ed = Ed.real
        return beta.real, Eu, Ed

    def diagonalization_error_message(self):
        return """The diagonalization failed in DORT. Several causes are possible:

- single scattering albedo > 1 in a layer. It is often due to a too large grain size (or too low stickiness
parameter, or too large polydispersity or too high frequency). Some emmodels (DMRT ShortRange, ...) that rely on the
Rayleigh/low-frequency assumption may produce unphysical single scattering albedo > 1. In this case, it is necessary to
reduce the grain size. If the phase_normalization option in DORT was desactivated (default is active), it is advised to
reactivate it.

- almost diagonal matrix. Such a matrix often arises in active mode when m_max is quite high. However it can
also arises in passive mode or with low m_max. To solve this issue  you can try to activate the
diagonalization_method="shur" option and if it does not work the more radical diagonalization_method="shur_forcedtriu".
These options are experimental, please report your results (both success and failure).
diagonalization_method="shur_forcedtriu" should become the default if success are reported. Alternatively you could
reduce the m_max option progressively but high values of m_max give more accurate results in active mode (but tends to
produce almost diagonal matrix).

For mass simulations, exceptions may be annoying, to avoid raising exception and return NaN as a result instead is
obtained by setting the option error_handling='nan'.

Note:: setting an option in DORT is obtained with make_model(..., "dort", rtsolver_options=dict(error_handling='nan')).
"""


def symmetrize_phase_matrix(A, m):
    n = A.shape[1] // 2
    newA = np.empty_like(A)

    if m == 0:
        npol = 2
        newA[:n, :n] = 0.5 * (A[:n, :n] + A[n:, n:])
        newA[n:, n:] = newA[:n, :n]
        newA[:n, n:] = 0.5 * (A[:n, n:] + A[n:, :n])
        newA[n:, :n] = newA[:n, n:]
    else:
        npol = 3
        for i in range(n):
            d0 = 1 if (i % npol) < 2 else -1
            for j in range(n):
                d = d0 if (j % npol) < 2 else -d0
                # alpha
                newA[i, j] = 0.5 * (A[i, j] + A[i + n, j + n] * d)
                newA[i + n, j + n] = d * A[i, j]
                # beta
                newA[i, j + n] = 0.5 * (A[i, j + n] + A[i + n, j] * d)
                newA[i + n, j] = d * newA[i, j + n]
    return newA


if numba:
    symmetrize_phase_matrix = numba.jit(symmetrize_phase_matrix)


class InterfaceProperties(object):
    def __init__(self, frequency, interfaces, substrate, permittivity, streams, m_max, npol):
        self.streams = streams

        nlayer = len(interfaces)

        self.Rtop_coh = dict()
        self.Rtop_diff = dict()
        self.Ttop_coh = dict()
        self.Ttop_diff = dict()
        self.Rbottom_coh = dict()
        self.Rbottom_diff = dict()
        self.Tbottom_coh = dict()
        self.Tbottom_diff = dict()
        self.full_weight = dict()

        for l in range(nlayer):
            eps_lm1 = permittivity[l - 1] if l > 0 else 1
            eps_l = permittivity[l]
            eps_lp1 = permittivity[l + 1] if l < nlayer - 1 else None

            # compute reflection coefficient between layer l and l - 1  UP
            # snow-snow UP
            self.Rtop_coh[l] = interfaces[l].specular_reflection_matrix(frequency, eps_l, eps_lm1, streams.mu[l], npol)

            self.Rtop_diff[l] = (
                interfaces[l].ft_even_diffuse_reflection_matrix(
                    frequency, eps_l, eps_lm1, streams.mu[l], streams.mu[l], m_max, npol
                )
                if hasattr(interfaces[l], "ft_even_diffuse_reflection_matrix")
                else smrt_matrix(0)
            )

            self.Rtop_diff[l] = normalize_diffuse_matrix(
                self.Rtop_diff[l], streams.mu[l], streams.mu[l], streams.weight[l]
            )

            # compute transmission coefficient between l and l - 1 UP
            # snow-snow or air UP
            self.Ttop_coh[l] = interfaces[l].coherent_transmission_matrix(
                frequency, eps_l, eps_lm1, streams.mu[l], npol
            )
            mu_t = streams.mu[l - 1] if l > 1 else streams.outmu
            self.Ttop_diff[l] = (
                interfaces[l].ft_even_diffuse_transmission_matrix(
                    frequency, eps_l, eps_lm1, mu_t, streams.mu[l], m_max, npol
                )
                * (eps_l.real / eps_lm1.real)
                if hasattr(interfaces[l], "ft_even_diffuse_transmission_matrix")
                else smrt_matrix(0)
            )

            self.Ttop_diff[l] = normalize_diffuse_matrix(self.Ttop_diff[l], mu_t, streams.mu[l], streams.weight[l])

            # compute transmission coefficient between l and l + 1  DOWN
            if l < nlayer - 1:
                # snow-snow DOWN
                self.Tbottom_coh[l] = interfaces[l + 1].coherent_transmission_matrix(
                    frequency, eps_l, eps_lp1, streams.mu[l], npol
                )

                self.Tbottom_diff[l] = (
                    interfaces[l + 1].ft_even_diffuse_transmission_matrix(
                        frequency, eps_l, eps_lp1, streams.mu[l + 1], streams.mu[l], m_max, npol
                    )
                    * (eps_l.real / eps_lp1.real)
                    if hasattr(interfaces[l + 1], "ft_even_diffuse_transmission_matrix")
                    else smrt_matrix(0)
                )
                self.Tbottom_diff[l] = normalize_diffuse_matrix(
                    self.Tbottom_diff[l], streams.mu[l + 1], streams.mu[l], streams.weight[l]
                )

            elif substrate is not None:
                # sub-snow
                self.Tbottom_coh[nlayer - 1] = substrate.emissivity_matrix(frequency, eps_l, streams.mu[l], npol)
                self.Tbottom_diff[nlayer - 1] = smrt_matrix(0)
            else:
                # sub-snow
                self.Tbottom_coh[nlayer - 1] = smrt_matrix(0)
                self.Tbottom_diff[nlayer - 1] = smrt_matrix(0)

            # compute reflection coefficient between l and l + 1  DOWN
            if l < nlayer - 1:
                # snow-snow DOWN
                self.Rbottom_coh[l] = interfaces[l + 1].specular_reflection_matrix(
                    frequency, eps_l, eps_lp1, streams.mu[l], npol
                )
                self.Rbottom_diff[l] = (
                    interfaces[l + 1].ft_even_diffuse_reflection_matrix(
                        frequency, eps_l, eps_lp1, streams.mu[l], streams.mu[l], m_max, npol
                    )
                    if hasattr(interfaces[l + 1], "ft_even_diffuse_reflection_matrix")
                    else smrt_matrix(0)
                )
                self.Rbottom_diff[l] = normalize_diffuse_matrix(
                    self.Rbottom_diff[l], streams.mu[l], streams.mu[l], streams.weight[l]
                )

            elif substrate is not None:
                # snow-substrate
                self.Rbottom_coh[l] = substrate.specular_reflection_matrix(frequency, eps_l, streams.mu[l], npol)

                self.Rbottom_diff[l] = (
                    substrate.ft_even_diffuse_reflection_matrix(
                        frequency, eps_l, streams.mu[l], streams.mu[l], m_max, npol
                    )
                    if hasattr(substrate, "ft_even_diffuse_reflection_matrix")
                    else smrt_matrix(0)
                )
                self.Rbottom_diff[l] = normalize_diffuse_matrix(
                    self.Rbottom_diff[l], streams.mu[l], streams.mu[l], streams.weight[l]
                )

            else:
                self.Rbottom_coh[l] = smrt_matrix(0)  # fully transparent substrate
                self.Rbottom_diff[l] = smrt_matrix(0)

        # air-snow DOWN
        self.Tbottom_coh[-1] = interfaces[0].coherent_transmission_matrix(
            frequency, 1, permittivity[0], streams.outmu, npol
        )

        self.Tbottom_diff[-1] = (
            interfaces[0].ft_even_diffuse_transmission_matrix(
                frequency, 1, permittivity[0], streams.mu[0], streams.outmu, m_max, npol
            )
            / permittivity[0].real
            if hasattr(interfaces[0], "ft_even_diffuse_transmission_matrix")
            else smrt_matrix(0)
        )
        self.Tbottom_diff[-1] = normalize_diffuse_matrix(
            self.Tbottom_diff[-1], streams.mu[0], streams.outmu, streams.outweight
        )

        # air-snow DOWN
        self.Rbottom_coh[-1] = interfaces[0].specular_reflection_matrix(
            frequency, 1, permittivity[0], streams.outmu, npol
        )
        self.Rbottom_diff[-1] = (
            interfaces[0].ft_even_diffuse_reflection_matrix(
                frequency, 1, permittivity[0], streams.outmu, streams.outmu, m_max, npol
            )
            if hasattr(interfaces[0], "ft_even_diffuse_reflection_matrix")
            else smrt_matrix(0)
        )
        self.Rbottom_diff[-1] = normalize_diffuse_matrix(
            self.Rbottom_diff[-1], streams.outmu, streams.outmu, streams.outweight
        )

    def reflection_top(self, l, m, compute_coherent_only, **kwargs):
        return InterfaceProperties.combine_coherent_diffuse_matrix(
            self.Rtop_coh[l], self.Rtop_diff[l], m, compute_coherent_only, **kwargs
        )

    def reflection_bottom(self, l, m, compute_coherent_only, **kwargs):
        return InterfaceProperties.combine_coherent_diffuse_matrix(
            self.Rbottom_coh[l], self.Rbottom_diff[l], m, compute_coherent_only, **kwargs
        )

    def transmission_top(self, l, m, compute_coherent_only, **kwargs):
        return InterfaceProperties.combine_coherent_diffuse_matrix(
            self.Ttop_coh[l], self.Ttop_diff[l], m, compute_coherent_only, **kwargs
        )

    def transmission_bottom(self, l, m, compute_coherent_only, **kwargs):
        return InterfaceProperties.combine_coherent_diffuse_matrix(
            self.Tbottom_coh[l], self.Tbottom_diff[l], m, compute_coherent_only, **kwargs
        )

    @staticmethod
    def combine_coherent_diffuse_matrix(coh, diff, m, compute_coherent_only, compress=True, auto_reduce_npol=True):
        mat_coh = coh.compress(mode=m, auto_reduce_npol=auto_reduce_npol) if compress else coh

        if (not compute_coherent_only) and (not is_equal_zero(diff)):
            # the coef comes from the integration of \int dphi' cos(m (phi-phi')) cos(n phi')
            # m=n=0 --> 2*np.pi
            # m=n > 1 --> np.pi
            if m == 0:
                coef = 2 * np.pi
                # npol = 2
            else:
                coef = np.pi  # the factor 2*np.pi comes from the integration of \int dphi
                # npol = 3

            mat_diff = diff.compress(mode=m, auto_reduce_npol=auto_reduce_npol) if compress else diff
            return coef * mat_diff + mat_coh
        else:
            return mat_coh


def normalize_diffuse_matrix(mat, mu_st, mu_i, weights):
    if is_equal_zero(mat):
        return mat

    if mat.mtype == "dense5":
        mat *= mu_i * weights  # the last dimension
        mat /= mu_st[:, np.newaxis]  # before the last dimension
    elif mat.mtype == "diagonal5":
        if mu_i is mu_st:
            mat *= weights
        else:
            mat *= mu_i * weights / mu_st  # the last dimension
    return mat
