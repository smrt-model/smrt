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
It is important not to set too low a value for n_max_stream. E.g. 32 is usually fine, 64 or 128 are better but simulations will be much slower.

"""


# Stdlib import
import math

# other import
import numpy as np
import scipy.special.orthogonal
import scipy.linalg
import scipy.interpolate
import scipy.sparse

# local import
from ..core.error import SMRTError
from ..core.result import Result
from ..core import lib
from ..core.lib import is_equal_zero


class DORT(object):
    """Discrete Ordinate and Eigenvalue Solver

        :param n_max_stream: number of stream in the most refringent layer
        :param m_max: number of mode (azimuth)

    """

    # this specifies which dimension this solver is able to deal with. Those not in this list must be managed by the called (Model object)
    # e.g. here, frequency, time, ... are not managed
    _broadcast_capability = {"theta_inc", "polarization_inc", "theta", "phi", "polarization"}

    def __init__(self, n_max_stream=32, m_max=2, stream_mode="most_refringent"):
        # """
        # :param n_max_stream: number of stream in the most refringent layer
        # :param m_max: number of mode (azimuth)

        # """
        self.n_max_stream = n_max_stream
        self.stream_mode = stream_mode
        self.m_max = m_max

    def solve(self, snowpack, emmodels, sensor, atmosphere=None):
        """solve the radiative transfer equation for a given snowpack, emmodels and sensor configuration.

"""
        try:
            len(self.sensor.phi)
        except Exception:
            pass
        else:
            raise Exception("phi as an array must be implemented")

        # all these assignements are for convenience, be carefull with // !
        self.snowpack = snowpack
        self.emmodels = emmodels
        self.sensor = sensor

        self.atmosphere = atmosphere

        self.nlayer = self.snowpack.nlayer

        self.thickness = self.snowpack.layer_thicknesses
        self.interfaces = self.snowpack.interfaces

        self.ke = [emmodel.ke for emmodel in emmodels]
        self.ft_even_phase = [emmodel.ft_even_phase for emmodel in emmodels]
        self.permittivity = np.array([emmodel.effective_permittivity() for emmodel in emmodels])

        if self.sensor.mode == 'P':
            self.temperature = [layer.temperature for layer in self.snowpack.layers]
        else:
            self.temperature = None

        m_max = self.m_max if self.sensor.mode == 'A' else 0  # force m_max=0 for passive microwave

        outmu, intensity = self.dort(m_max=m_max)

        #
        # reshape the intensity vector to unpack the theta, polarization
        #
        pola = ['V', 'H'] if self.sensor.mode == 'P' else ['V', 'H', 'U']
        npol = len(pola)

        # reshape the first dimension in two dimensions (theta, pola)
        intensity = intensity.reshape([intensity.shape[0]//npol, npol]+list(intensity.shape[1:]))

        mu = np.cos(sensor.theta)
        if min(mu) < min(outmu) or max(mu) > max(outmu):
            raise SMRTError("viewing zenith angle is outside the range of stream angles computed by DORT. Increase the number of streams or change your viewing zenith angle range. In the future it will be possible to explicitly force the extrapolation.")

        # reverse is necessary for "old" scipy version
        intfct = scipy.interpolate.interp1d(outmu[::-1], intensity[::-1, :, ...], axis=0)  # could use fill_value to be smart about extrapolation, but it's safer to return NaN (default)

        # it seems there is a bug in scipy at least when checking the boundary, mu must be sorted, outmu does not !
        i = np.argsort(mu)

        # original that should work: intensity = intfct(mu)
        intensity = intfct(mu[i])[np.argsort(i)]  # mu[i] sort mu, and [np.argsort(i)] put in back

        if sensor.mode == 'A':
            # reshape the outer/first dimension in two dimensions (theta_inc, pola_inc)
            intensity = intensity.reshape(list(intensity.shape[:-1])+[intensity.shape[-1]//npol, npol])
        #  describe the results list of (dimension name, dimension array of value)
        coords = [('theta', sensor.theta_deg), ('polarization', pola)]
        if sensor.mode == 'A':
            coords = [('theta_inc', sensor.theta_inc_deg), ('polarization_inc', pola)] + coords

        return Result(intensity, coords)

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

        permittivity_substrate = self.snowpack.substrate.permittivity(self.sensor.frequency) if self.snowpack.substrate is not None else None

        n_stream, mu, weight, outmu, outweight, \
        n_stream_substrate = compute_stream(self.n_max_stream, self.permittivity, permittivity_substrate, mode=self.stream_mode)

        #
        # compute the incident intensity array depending on the sensor

        intensity_0, intensity_higher = self.prepare_intensity_array(outmu, outweight)  # TODO Ghi: make an iterator

        #
        # need to compute coherent wave propagation ?

        compute_coherent_only = self.sensor.mode == 'A'

        # inform the EM model how many modes will be needed. This is only used for optimization purpose.
        # alternative would be to start computing by m_max
        for emmodel in self.emmodels:
            if hasattr(emmodel, "set_max_mode"):
                emmodel.set_max_mode(m_max)

        #
        # compute the modes

        npol = 3 if self.sensor.mode == 'A' else 2

        for m in range(0, m_max+1):  # TODO Ghi: parallelizable (be carefull with caching in the callables)
            if m == 0:
                intensity_down_m = intensity_0
            else:
                intensity_down_m = intensity_higher

            # compute the upwelling intensity for mode m
            intensity_up_m = self.dort_modem_banded(m, n_stream, mu, weight, outmu, n_stream_substrate, intensity_down_m, special_return=special_return)

            if special_return:  # debugage
                return intensity_up_m

            if compute_coherent_only:
                # substrate the coherent contribution
                intensity_up_m -= self.dort_modem_banded(m, n_stream, mu, weight, outmu, n_stream_substrate, intensity_down_m, compute_coherent_only=True)

            # if self.sensor.mode == 'A': print("res mod=", m, intensity_up_m[0:3, 0:3])
            # reconstruct the intensity
            if m == 0:
                intensity_up = extend_2pol_npol(intensity_up_m, npol)
            else:
                intensity_up[0::npol] += intensity_up_m[0::npol] * np.cos(m*self.sensor.phi)  # TODO Ghi: deals with an array of phi
                intensity_up[1::npol] += intensity_up_m[1::npol] * np.cos(m*self.sensor.phi)  # TODO Ghi: deals with an array of phi
                intensity_up[2::npol] += intensity_up_m[2::npol] * np.sin(m*self.sensor.phi)  # TODO Ghi: deals with an array of phi

                # TODO: implement a convergence test if we want to avoid long computation when self.m_max is too high for the phase function.

        if self.atmosphere is not None:
            intensity_up = self.atmosphere.tbup(self.sensor.frequency, outmu, npol) + \
                        self.atmosphere.trans(self.sensor.frequency, outmu, npol) * intensity_up

        return outmu, intensity_up

    def prepare_intensity_array(self, outmu, outweight):

        if self.sensor.mode == 'A':
            # send a direct beam

            # incident angle at a given angle
            # use interpolation to get the based effective angle

            intensity_0 = np.zeros((2*len(outmu), 2*len(self.sensor.theta_inc)))  # 2 is for the two polarizations

            j0 = 0
            for theta in self.sensor.theta_inc:
                mu_inc = math.cos(theta)
                i0 = np.searchsorted(-outmu, -mu_inc)

                if i0 == 0 or i0==len(outmu):
                    if i0 == len(outmu):
                        i0 -= 1
                    for ipol in [0, 1]:
                        intensity_0[2*i0 + ipol, j0] = 1.0 / (2 * math.pi * outweight[i0])
                        j0 += 1
                else:
                    i1 = i0 - 1
                    alpha = (outmu[i0]-mu_inc)/(outmu[i0]-outmu[i1])

                    delta_0 = (1 - alpha) / (2 * math.pi * outweight[i0])
                    delta_1 = alpha / (2 * math.pi * outweight[i1])

                    for ipol in [0, 1]:
                        intensity_0[2*i0 + ipol, j0] = delta_0
                        intensity_0[2*i1 + ipol, j0] = delta_1
                        j0 += 1

            intensity_higher = 2 * extend_2pol_npol(intensity_0, 3)

        elif self.sensor.mode == 'P':

            npol = 2

            if self.atmosphere is not None:

                # incident radiation is a function of frequency and incidence angle
                # assume azimuthally symmetric
                intensity_0 = self.atmosphere.tbdown(self.sensor.frequency, outmu, npol)[:, np.newaxis]
                intensity_higher = np.zeros_like(intensity_0)

            else:
                intensity_0 = np.zeros((len(outmu)*npol, 1))
                intensity_higher = intensity_0
                intensity_0.flags.writeable = False  # make immutable
                intensity_higher.flags.writeable = False  # make immutable

        else:
            raise SMRTError("Unknow sensor mode")

        return intensity_0, intensity_higher

    def dort_modem_banded(self, m, n_stream, mu, weight, outmu, n_stream_substrate, intensity_down_m, compute_coherent_only=False, special_return=False):

        n_stream0 = len(outmu)  # number of streams in the air

        # Index convention
        # for phase, Ke, and R matrix pola must be the fast index, then stream, then +-
        # order of the boundary condition in row: layer, equation, stream+/stream-, pola
        # order of the boundary condition in column: layer, -/+, etc

        npol = 2 if m == 0 else 3

        # indexes of the columns
        jl = 2 * (np.cumsum(n_stream)-n_stream) * npol

        # indexes of the rows: start Eq 19, then Eq TOP then Eq BOTTOM, then Eq 22
        il_top = 2 * (np.cumsum(n_stream)-n_stream) * npol
        il_bottom = il_top + n_stream * npol

        nboundary = sum(n_stream) * 2 * npol
        nband = 3 * npol * np.max(n_stream)  # each layer appears in 3 blocks
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

        for l in range(0, self.nlayer):
            nsl = n_stream[l]  # number of streams in layer l
            nslnpol = nsl * npol  # number of streams * npol in layer l
            nsl2npol = 2 * nslnpol    # number of eigenvalues in layer l (should be 2*npol*n_stream)
            nslm1npol = (n_stream[l - 1] * npol) if l > 0 else (n_stream0 * npol)  # number of streams * npol in the layer l - 1 (lm1)
            nslp1npol = (n_stream[l + 1] * npol) if l < self.nlayer - 1 else (n_stream_substrate * npol)  # number of streams * nplo in the layer l + 1 (lp1)

            # solve the eigenvalue problem for layer l
            # TODO: deal with the case phase=0
            if self.ft_even_phase is None or compute_coherent_only:
                ft_even_phase_l = None
            else:
                ft_even_phase_l = self.ft_even_phase[l]

            beta, Eu, Ed = solve_eigenvalue_problem(m, self.ke[l], ft_even_phase_l, mu[l, 0:nsl], weight[l, 0:nsl])

            # deduce the transmittance through the layers
            transt = scipy.sparse.diags(np.exp(-np.maximum(beta, 0) * self.thickness[l]), 0)  # positive beta, reference at the bottom
            transb = scipy.sparse.diags(np.exp(np.minimum(beta, 0) * self.thickness[l]), 0)   # negative beta, reference at the top

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

            # compute reflection coefficient between l and l - 1
            if l == 0:
                epslm1 = 1
                # save these matrix to compute the emerging intensity at the end
                Eu_0 = Eu
                transt_0 = transt
            else:
                epslm1 = self.permittivity[l - 1]

            Rtop_l = fix_matrix(self.interfaces[l].specular_reflection_matrix(self.sensor.frequency, self.permittivity[l], epslm1,
                                                                 mu[l, 0:nsl], npol))  # snow-snow

            # fill the matrix
            todiag(bBC, (il_topl, j), (Ed - Rtop_l * Eu) * transt)  # this line perform matrix multiplication between Rtop_l and Eu. Make sure that reflection_matrix return matrix!

            if debug_compute_BC:
                BC[il_topl:il_topl + nslnpol, j:j+nsl2npol] = (Ed - Rtop_l * Eu) * transt   # a mettre en (l,l), theta<0 et * transt  # a mettre en (l,l)

            if l < self.nlayer - 1:
                ns_ = min(nslnpol, nslp1npol)
                Tbottom_lp1 = fix_matrix(self.interfaces[l].coherent_transmission_matrix(self.sensor.frequency, self.permittivity[l], self.permittivity[l + 1],
                                                                              mu[l, 0:(ns_//npol)], npol))  # snow-snow
                todiag(bBC, (il_top[l + 1], j), - Tbottom_lp1 * Ed[0:ns_, :] * transb)
                if debug_compute_BC:
                    BC[il_top[l + 1]:il_top[l + 1]+ns_, j:j+nsl2npol] = - Tbottom_lp1 * Ed[0:ns_, :] * transb  # a mettre en (l + 1,l)

            # fill the vector
            if m == 0 and self.temperature is not None and self.temperature[l] > 0:
                if is_equal_zero(Rtop_l):
                    b[il_topl:il_topl + nslnpol, :] -= self.temperature[l]  # a mettre en (l)
                else:
                    b[il_topl:il_topl + nslnpol, :] -= ((1.0 - muleye(Rtop_l)) * self.temperature[l])[:, np.newaxis]  # a mettre en (l)
                # the muleye comes from the isotropic emission of the black body

                if l < self.nlayer - 1 and self.temperature[l] > 0:
                    b[il_top[l + 1]:il_top[l + 1]+ns_, :] += (muleye(Tbottom_lp1) * self.temperature[l])[:, np.newaxis]     # a mettre en (l + 1)

            if l == 0:  # Air-snow interface
                Tbottom_air_down = fix_matrix(self.interfaces[l].coherent_transmission_matrix(self.sensor.frequency, 1, self.permittivity[l],
                                                                                   outmu, npol))

                b[il_topl:il_topl+n_stream0*npol, :] += Tbottom_air_down * intensity_down_m

            # -------------------------------------------------------------------------------
            # Eq 18 & 22 BOTTOM of layer l

            # compute reflection coefficient between l and l - 1
            if l < self.nlayer-1:
                Rbottom_l = fix_matrix(self.interfaces[l].specular_reflection_matrix(self.sensor.frequency, self.permittivity[l], self.permittivity[l + 1], mu[l, 0:nsl], npol))  # snow-snow
            elif self.snowpack.substrate is not None:
                Rbottom_l = fix_matrix(self.snowpack.substrate.specular_reflection_matrix(self.sensor.frequency, self.permittivity[l], mu[l, 0:nsl], npol))  # snow-sub
                if not compute_coherent_only and hasattr(self.snowpack.substrate, "ft_even_diffuse_reflection_matrix"):
                    full_weight_l = np.repeat(weight[l, 0:nsl], npol)    # could be cached (per layer) because same for each mode
                    Rbottom_l += fix_matrix(self.snowpack.substrate.ft_even_diffuse_reflection_matrix(m, self.sensor.frequency, self.permittivity[l], mu[l, 0:nsl], npol)) * full_weight_l  # snow-sub
            else:
                Rbottom_l = 0  # fully absorbant substrate

            # fill the matrix
            todiag(bBC, (il_bottoml, j), (Eu - Rbottom_l * Ed) * transb)
            if debug_compute_BC:
                BC[il_bottoml:il_bottoml+nslnpol, j:j+nsl2npol] = (Eu - Rbottom_l * Ed) * transb  # a mettre en (l,l), theta >0

            if l > 0:
                ns_ = min(nslnpol, nslm1npol)
                Ttop_lm1 = fix_matrix(self.interfaces[l].coherent_transmission_matrix(self.sensor.frequency, self.permittivity[l], self.permittivity[l - 1], mu[l, 0:(ns_//npol)], npol))  # snow-snow
                todiag(bBC, (il_bottom[l - 1], j), -Ttop_lm1 * Eu[0:ns_, :] * transt)
                if debug_compute_BC:
                    BC[il_bottom[l - 1]:il_bottom[l - 1]+ns_, j:j+nsl2npol] = -Ttop_lm1 * Eu[0:ns_, :] * transt  # a mettre en (l - 1)

            # fill the vector
            if m == 0 and self.temperature is not None and self.temperature[l] > 0:
                if Rbottom_l is 0:
                    b[il_bottoml:il_bottoml+nslnpol, :] -= self.temperature[l]   # a mettre en (l)
                else:
                    b[il_bottoml:il_bottoml+nslnpol, :] -= ((1.0 - muleye(Rbottom_l)) * self.temperature[l])[:, np.newaxis]  # a mettre en (l)
                if l > 0:
                    b[il_bottom[l - 1]:il_bottom[l - 1]+ns_, :] += (muleye(Ttop_lm1) * self.temperature[l])[:, np.newaxis]  # a mettre en (l - 1)

            if m == 0 and l == self.nlayer-1 and self.snowpack.substrate is not None and \
                self.snowpack.substrate.temperature is not None and self.temperature is not None:
                ####Rtop_sub = self.interfaces[l].specular_reflection_matrix(npol, sensor.frequency, substrate.permittivity, permittivity[l], mu[l, 0:nsl], compute_coherent_only)  # sub-snow
                ###raise Exception("finish the implementation here")
                ###Rtop_sub = self.snowpack.substrate.emission_matrix(self.sensor.frequency, self.permittivity[l], mu[l, 0:nsl], compute_coherent_only)  # sub-snow
                Ttop_sub = fix_matrix(self.snowpack.substrate.emissivity_matrix(self.sensor.frequency, self.permittivity[l], mu[l, 0:nsl], npol))  # sub-snow
                b[il_bottoml:il_bottoml+nslnpol, :] += (muleye(Ttop_sub) * self.snowpack.substrate.temperature)[:, np.newaxis]

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


        #x = scipy.linalg.solve(BC, b, overwrite_a=True, overwrite_b=False)
        x = scipy.linalg.solve_banded((nband, nband), bBC, b, overwrite_ab=True, overwrite_b=True)
        x = np.matrix(x)

        # #  ! calculate the intensity emerging from the snowpack
        l = 0
        j = jl[l]  # should be 0
        nsl2npol = 2 * n_stream[l] * npol
        I1up_m = Eu_0 * transt_0 * x[j:j+nsl2npol, :]

        if m == 0 and self.temperature is not None and self.temperature[0] > 0:
            I1up_m += self.temperature[0]  # just under the interface

        Rbottom_air_down = fix_matrix(self.interfaces[0].specular_reflection_matrix(self.sensor.frequency, 1, self.permittivity[0],
                                                                         outmu, npol))
        Ttop_0 = fix_matrix(self.interfaces[0].coherent_transmission_matrix(self.sensor.frequency, self.permittivity[0],
                                                                 1, mu[0, 0:n_stream[0]], npol))  # snow-air

        I0up_m = Rbottom_air_down * np.matrix(intensity_down_m) + (Ttop_0 * I1up_m)[:npol*n_stream0]

        return np.array(I0up_m).squeeze()


def fix_matrix(x):
    #"""fix the change of diagonal object"""
    if isinstance(x, lib.diag):
        return x.as_dia_matrix()
    else:
        return x


def muleye(x):
    #"""multiply x * 1v """
    if isinstance(x, scipy.sparse.dia_matrix) or isinstance(x, np.matrix):
        return x.sum(axis=1).A1
    else:
        return x


def todiag(bmat, ij, dmat):
    #"""insert the small dense dmat matrix in the diagonal bmat matrix"""
    oi, oj = ij

    u = (bmat.shape[0]-1) // 2
    l = u  # special case here

    n, m = dmat.shape
    df = np.array(dmat).flatten('C')

    for j in range(0, min(u+1, m)):
        ldiag = min(n, m-j)
        bmat[u+oi-oj-j, j+oj:j+oj+ldiag] = df[j::m+1][:ldiag]

    for i in range(1, min(l + 1, n)):
        ldiag = min(n-i, m)
        bmat[u+oi-oj+i, 0+oj:0+oj+ldiag] = df[i*m::m+1][:ldiag]


def extend_2pol_npol(x, npol):

    if npol == 2:
        return x

    if scipy.sparse.isspmatrix_dia(x):
        y = scipy.sparse.diags(extend_2pol_npol(x.diagonal(), npol))
    elif len(x.shape) == 1:
        y = np.zeros(len(x)//2*npol)
        y[0::npol] = x[0::2]
        y[1::npol] = x[1::2]
    elif len(x.shape) == 2:
        y = np.zeros((x.shape[0]//2*npol, x.shape[1]//2*npol))
        y[0::npol, 0::npol] = x[0::2, 0::2]
        y[0::npol, 1::npol] = x[0::2, 1::2]
        y[1::npol, 0::npol] = x[1::2, 0::2]
        y[1::npol, 1::npol] = x[1::2, 1::2]
    else:
        raise SMRTError("should never be here")

    return y


def solve_eigenvalue_problem(m, ke, ft_even_phase, mu, weight):
    # """solve the homogeneous equation for a single layer and return eigne value and eigen vector

    # :param m: mode
    # :param Ke: extinction coefficient of the layer for mode m
    # :param ft_even_phase: ft_even_phase function of the layer for mode m
    # :param mu: cosines
    # :param weight: weights

    # :returns: alpha, E, Q
    #"""
    nsk = len(mu)

    npol = 2 if m == 0 else 3

    n = npol*nsk

    # this coefficient comme from the 1/4pi normalization of the RT equation
    coef = 0.5 if m == 0 else 0.25

    # compute invmu
    invmu = 1.0 / mu
    invmu = np.repeat(invmu, npol)
    invmu = np.concatenate((invmu, -invmu))
    mu = np.concatenate((mu, -mu))

    if ft_even_phase is None:
        # special case (important for the coherency)
        beta = invmu * np.repeat(ke(mu), npol)
        E = np.eye(2*n, 2*n)  # TODO: test with a sparse matrix if more performant
    else:
        # calculate the A matrix. Eq (12)

        A = ft_even_phase(m, mu, mu)

        if A is 0:
            beta = invmu * np.repeat(ke(mu), npol)
            E = np.eye(2*n, 2*n)  # TODO: test with a sparse matrix if more performant
        else:
            # solve the eigen value problem
            weight = np.tile(np.repeat(-coef * weight, npol), 2)    # could be cached (per layer) because same for each mode
            A *= weight[np.newaxis, :]
            A[np.diag_indices(2*n)] += np.repeat(ke(mu), npol)
            A = invmu[:, np.newaxis] * A

            # diagonalise the matrix. Eq (13)
            try:
                beta, E = scipy.linalg.eig(A, overwrite_a=True)
            except scipy.linalg.LinAlgError:
                diagonalization_failed = True
                reason = "eig method"
            else:
                diagonalization_failed = (not np.allclose(beta.imag, 0)) or (not np.allclose(E.imag, 0))
                if diagonalization_failed:
                    print(np.allclose(beta.imag, 0), np.allclose(E.imag, 0))
                    print(np.max(np.abs(E.imag)))
                reason = "not close"

            if diagonalization_failed:

                raise SMRTError("""The diagonalization failed in DORT which is possibly caused by single scattering albedo larger than 1.
It is often due to grain size too large (or too low stickiness parameter) to respect the Rayleigh/low-frequency assumption required by some ememodel (DMRT ShortRange, IBA, ...)"
It is recommended to reduce the size of the bigger grains.""")

            #np.testing.assert_allclose(beta.imag, 0, atol=np.linalg.norm(beta)*1e-5)

            if np.iscomplexobj(E):
                mask = abs(E.imag)>np.linalg.norm(E)*1e-5
                if np.any(mask):
                    print(np.any(mask, axis=1))
                    print(beta[np.any(mask, axis=1)])
                    print(beta)

            #np.testing.assert_allclose(E.imag, 0, atol=np.linalg.norm(E)*1e-5)

            beta = beta.real
            #E = E.real

            # get the positive and negative beta
            # this should be improve a the mathematical level because there is no need to solve
            # the eigenvalue for + and - well according to inverse optical path equivalent,
            # the + and - should be equal

            # debub only
            if True:
                idx2 = np.argsort(beta)
                idx2[0:n] = idx2[0:n][::-1]
                beta = beta[idx2]
                E = E[:, idx2]

    Eu = E[0:n, :]  # upwelling
    Ed = E[n:, :]  # downwelling

    return beta, Eu, Ed
    # !-----------------------------------------------------------------------------!


def compute_stream(n_max_stream, permittivity, permittivity_substrate, mode="most_refringent"):
    #     """Compute the optimal angles of each layer. Use for this a Gauss-Legendre quadrature for the most refringent layer and
    # use Snell-law to prograpate the direction in the other layers takig care of the total reflection.

    #     :param n_max_stream: number of stream
    #     :param permittivity: permittivity of each layer
    #     :type permittivity: ndarray
    #     :returns: mu, weight, outmu
    # """

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

        n_stream = scipy.optimize.brentq(n_max_stream, 2*n_max_stream)
        mu_most_refringent, weight_most_refringent = gaussquad(n_stream)

    else:
        raise Runtime("Unknow mode to compute the number of stream")

    nlayer = len(permittivity)
    mu = np.zeros((nlayer, n_max_stream), dtype=np.float64)

    #  calculate the nodes and weights of all the other layers

    # calculate real part of the index. It is an approximation. See for instance "Semiconductor Physics, Quantum Electronics & Optoelectronics. 2001. V. 4, N 3. P. 214-218."
    real_index = np.real(np.sqrt(permittivity[k_most_refringent] / permittivity[:]))

    # calculate the angles (=node)
    relsin = real_index[:, np.newaxis] * np.sqrt(1 - mu_most_refringent[np.newaxis, :]**2)

    real_reflection = relsin < 1  # mask where real reflection occurs
    mu[real_reflection] = np.sqrt(1-relsin[real_reflection]**2)

    # calculate the number of stream per layer
    n_stream = np.sum(real_reflection, axis=1)

    assert(all(n_stream > 2))

    # calculate the weight ("a" in Y-Q Jin)
    # weight(1,k)=1-0.5*(mu(1,k)+mu(2,k))
    # weight(nsk,k)=0.5*(mu(nsk-1,k)+mu(nsk,k))
    # weight(2:nsk-1,k)=0.5*(mu(1:nsk-2,k)-mu(3:nsk,k))

    weight = np.empty_like(mu)
    weight[:, 0] = 1 - 0.5*(mu[:, 0] + mu[:, 1])
    for k in range(nlayer):   # TODO: parallel
        #if True or k != k_most_refringent:
        nsk = n_stream[k]
        weight[k, nsk-1] = 0.5*(mu[k, nsk-2] + mu[k, nsk-1])
        weight[k, 1:nsk-1] = 0.5*(mu[k, 0:nsk-2] - mu[k, 2:nsk])  # TODO PERF: voir si pas mieux de faire un diff 2 ou de mettre avant et faire le calcul
        #else:
        #    weight[k_most_refringent, :] = weight_most_refringent

    # ### calculate the angles (=node) in the air
    # real_index = np.real(np.sqrt(permittivity[0]/1.0))
    # relsin = real_index * np.sqrt(1 - mu[0, :]**2)

    # real_reflection = relsin < 1
    # outmu = np.sqrt(1 - relsin[real_reflection]**2)

    relsin = real_index_air * np.sqrt(1 - mu_most_refringent[:]**2)

    real_reflection = relsin < 1
    outmu = np.sqrt(1 - relsin[real_reflection]**2)

    outweight = np.empty_like(outmu)
    outweight[0] = 1 - 0.5*(outmu[0] + outmu[1])
    outweight[-1] = 0.5*(outmu[-2] + outmu[-1])
    outweight[1:-1] = 0.5*(outmu[0:-2] - outmu[2:])

    # compute the number of stream in the substrate
    if permittivity_substrate is None:
        n_stream_substrate = n_stream[-1]  # same as last layer
    else:
        real_index = np.real(np.sqrt(permittivity_substrate / permittivity[-1]))

        # calculate the angles (=node) in the substrate
        relsin = real_index * np.sqrt(1 - mu_most_refringent[:]**2)
        n_stream_substrate = np.sum (relsin < 1)   # count where real reflection occurs

    return n_stream, mu, abs(weight), outmu, outweight, n_stream_substrate


def gaussquad(n):
    #     """return the gauss-legendre roots and weight, only the positive roots are return.

    #     :param n: number of (positive) points in the quadrature. Must be larger than 2
    # """
    assert n >= 2

    mu, weight = scipy.special.orthogonal.p_roots(2*n)

    mu = mu[-1:n-1:-1]
    weight = weight[-1:n-1:-1]

    return mu, weight
