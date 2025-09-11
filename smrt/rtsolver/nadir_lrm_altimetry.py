"""
Compute waveforms as measured by Low Rate Mode altimeters (e.g. ENVISAT) for the given snowpack and sensor (or complex terrain soon).

The implementation is based on Adams and Brown 1998 and Lacroix et al. 2008. Both models differ in the specific choices for the
scattering and backscatter of the interface, but are similar in the way the waveform is calculated, which concerns the solver here.

Approximations:
    - Backscatter is computed assuming only first order scattering. The propagation is then simply governed by extinction.
    - Near nadir / small angle approximation: to compute delay, the paths in the snow are along the z-axis. Off-nadir delay is neglected.
      This error is likely to be small (except for very deep penetration).
    - Do not account for specular refelction (to be implemented).

Note:
    With this RT solver, if using Geometrical Optics for rough surface/interface modeling, it is strongly advised to use
    :py:func:`smrt.interface.geometrical_optics_backscatter` instead of :py:func:`smrt.interface.geometrical_optics` for
    the reason explained in the documentation of those modules.

Usage:
    Basic usage with default settings and iba emmodel:
        >>> altimodel = make_model('iba', "nadir_lrm_altimetry")

    Return individual contributions:
        >>> altimodel = make_model('iba', "nadir_lrm_altimetry", rtsolver_options={return_contributions=True})
"""

import numpy as np
import scipy.signal

from smrt.core.globalconstants import C_SPEED
from smrt.core.error import SMRTError
from smrt.core.result import AltimetryResult
from smrt.rtsolver.waveform_model import Brown1977
from smrt.interface.flat import Flat

import xarray as xr


class NadirLRMAltimetry(object):
    """
    Implement the Nadir LRM Mode Altimetry RT solver.

    Args:
        oversampling_time: integer number defining the number of subgates used for the computation in each altimeter gate.
            This is equivalent to multiply the bandwidth by this number. It is used to perform more accurate computation.
        return_oversampled: by default the backscatter is returned for each gate. If set to True, the oversampled waveform
            is returned instead. See the 'oversampling' argument.
        return_contributions: return "volume", "surface" and "interface" backscatter contributions in addition to the "total" backscatter.
        compute_coherent_reflection: if True (default), compute the coherent reflection according to Fung and Eom, 1983.
        skip_pfs_convolution: return the vertical backscatter without the convolution by the PFS, if set to True.
        theta_inc_sampling: number of subdivisions used to calculate the incidence angular variations of surface and inteface
            backscatter (the higher the better but the more computationnaly expensive). Note
            that the subdivisions are irregular in incidence angle but correspond to annulii of equi-duration. This number
            must be a true divider of the number of gates.
        error_handling: If set to "exception" (the default), raise an exception in case of error, stopping the code.
            If set to "nan", return a nan, so the calculation can continue, but the result is of course unusuable and
            the error message is not accessible. This is only recommended for long simulations that sometimes produce an error.
    """

    # this specifies which dimension this solver is able to deal with. Those not in this list must be managed by the called (Model object)
    # e.g. here, frequency, time, ... are not managed
    _broadcast_capability = {}  # "theta_inc", "polarization_inc", "theta", "phi", "polarization"}

    def __init__(
        self,
        waveform_model=None,
        oversampling_time=10,
        return_oversampled=False,
        skip_pfs_convolution=False,
        return_contributions=False,
        compute_coherent_reflection=True,
        theta_inc_sampling=8,
        error_handling="exception",
    ):
        # """

        # """

        super().__init__()

        self.waveform_model_class = waveform_model if waveform_model is not None else Brown1977
        self.error_handling = error_handling
        self.oversampling = oversampling_time
        self.return_contributions = return_contributions
        self.compute_coherent_reflection = compute_coherent_reflection
        self.return_oversampled = return_oversampled
        self.skip_pfs_convolution = skip_pfs_convolution
        self.theta_inc_sampling = theta_inc_sampling

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
        if sensor.theta_inc != 0:
            raise SMRTError("This solver is for nadir looking altimeter only")
        assert atmosphere is None

        self.snowpack = snowpack
        self.emmodels = emmodels
        self.sensor = sensor  # setting this make this solver incompatible with // computing
        self.waveform_model = self.waveform_model_class(sensor)

        # do we compute the surface and interface backscatter at nadir only or for several incidence angles ?
        if self.theta_inc_sampling > 1:
            # regular sampling in time
            if self.sensor.ngate % self.theta_inc_sampling != 0:
                raise SMRTError("The number 'theta_inc_sampling' must be a true divider of the number of gates.")
            t_inc_sample = np.linspace(0, self.sensor.ngate / self.sensor.pulse_bandwidth, self.theta_inc_sampling + 1)
            # cosine is adjacent divided by hypothenuse. This neglects the Earth sphericity.
            mu_i = 1.0 / (1.0 + C_SPEED * t_inc_sample / sensor.altitude)
        else:
            # nadir only
            mu_i = 1.0
            t_inc_sample = [0]

        # compute the vertical backscatter
        self.z_gate, _ = self.gate_depth()

        backscatter = self.vertical_scattering_distribution(
            mu_i=mu_i, return_contributions=self.return_contributions or (self.theta_inc_sampling > 1)
        )

        # compute the t_gate, taking into account the oversampling but not the shift
        ngate = self.sensor.ngate
        t_gate = np.arange(0, ngate * self.oversampling) / (self.sensor.pulse_bandwidth * self.oversampling)

        # need of padding ? if yes, pad the end with zeros
        if backscatter.shape[-1] < len(t_gate):
            backscatter = np.append(
                backscatter, np.zeros(backscatter.shape[:-1] + (len(t_gate) - backscatter.shape[-1],)), axis=-1
            )

        # compute the convolution with pfs and ptr if requested
        if self.skip_pfs_convolution or (self.waveform_model is None):
            waveform = backscatter
        else:
            waveform = self.convolve_with_PFS_PTR_PDF(t_gate, backscatter, t_inc_sample)

        # limit the waveform to the number of gates
        if waveform.shape[-1] > len(t_gate):
            waveform = waveform[..., : len(t_gate)]

        # downsample
        if self.oversampling > 1 and not self.return_oversampled:
            t_gate = t_gate[:: self.oversampling]
            self.z_gate = self.z_gate[:: self.oversampling]
            newshape = list(waveform.shape[0:-1]) + [
                -1,
                self.oversampling,
            ]  # split the last dimension into two, to agregate the subgates
            waveform = np.mean(waveform.reshape(newshape), axis=-1)

        # prepare the output in the Result object
        theta_inc_deg = [0]
        coords = [
            ("delay", t_gate - self.sensor.nominal_gate / self.sensor.pulse_bandwidth),
            ("theta_inc", theta_inc_deg),
            ("theta", theta_inc_deg),
        ]

        if self.return_contributions:
            total = np.sum(waveform, axis=0)  # compute the total

            waveform = np.append(waveform, total[None, :], axis=0)  # add the total
            coords = [("contribution", ["surface", "interfaces", "volume", "total"])] + coords

        res = xr.DataArray(
            waveform[..., None, None], coords=coords
        )  # convert to xarray, adding two empty dim for theta_inc and theta

        # add useful coordinates
        res = res.assign_coords(gate=res.delay * sensor.pulse_bandwidth + sensor.nominal_gate)
        # create the altimetry result
        res = AltimetryResult(res)

        if len(self.z_gate) >= len(t_gate):
            # shorten
            self.z_gate = self.z_gate[0 : len(t_gate)]
        else:
            # extend with nan's
            self.z_gate = np.append(self.z_gate, np.full(len(t_gate) - len(self.z_gate), np.nan))

        # that's a hack... we should make an AltimetryResult class
        res.z_gate = xr.DataArray(self.z_gate, coords=[res.delay])
        return res

    def convolve_with_PFS_PTR_PDF(self, t_gate, backscatter, t_inc_sample):
        # take into account the PFS (flat surface) and PTR (point target response=pulse width effect)
        sigma_surface = getattr(self.snowpack, "sigma_surface", 0)
        surface_slope = getattr(self.snowpack, "surface_slope", 0)

        if hasattr(self.waveform_model, "PFS_PTR_PDF") and self.theta_inc_sampling == 1:
            # good, we have PFS and PTR already convoluted, fast pathway
            pfs_ptr_pdf = self.waveform_model.PFS_PTR_PDF(
                t_gate, sigma_surface=sigma_surface, surface_slope=surface_slope
            )

            def do_convolve_by_PFS_PTR_PDF(backscatter):
                return scipy.signal.convolve(pfs_ptr_pdf, backscatter, mode="full")

            if self.return_contributions:
                # this is faster, as we don't need to take into account for the incidence variations
                return np.apply_along_axis(do_convolve_by_PFS_PTR_PDF, axis=-1, arr=backscatter)
            else:
                # this is even faster, as we don't need to separate the different contributions.
                return do_convolve_by_PFS_PTR_PDF(backscatter)

        else:  # slow pathway: a more complex situation, we must compute and combine the PFS and backscatter separately
            # then the (PTR and PDF)

            # start with building the PTR
            if (self.sensor.pulse_sigma > 0) or (sigma_surface > 0):
                sigma_c = np.sqrt(self.sensor.pulse_sigma**2 + (2 * sigma_surface / C_SPEED) ** 2)

                # restrict t_gate to 5 sigma and compute the positive and negative values
                i = min(np.searchsorted(t_gate, 5 * sigma_c), len(t_gate) - 1)
                sym_t_gate = np.concatenate((-t_gate[i:0:-1], t_gate[0:i]))

                ptr_pdf = np.exp(-0.5 * (sym_t_gate / sigma_c)**2)
                ptr_pdf /= np.sum(ptr_pdf) * self.sensor.pulse_bandwidth
                #  widening is necessary to get the tail correct after the convolution
                extended_t_gate = t_gate[0] + (t_gate[-1] - t_gate[0]) * np.arange(len(t_gate) + i) / len(t_gate)
            else:
                extended_t_gate = t_gate

            # first compute the PFS
            if hasattr(self.waveform_model, "PFS"):
                # at least we have an analytical PFS
                pfs = self.waveform_model.PFS(extended_t_gate, surface_slope=surface_slope)
            else:
                # we've no PFS, we need numerical integration
                pfs = self.PFS_numerical(extended_t_gate)

            # combine the PFS with the backscatters
            #
            # split the backscatter contributions:
            nmu = len(t_inc_sample)
            backscatter_surface, backscatter_interfaces, backscatter_volume = (
                backscatter[0:nmu, 0],
                backscatter[nmu : 2 * nmu],
                backscatter[-1],
            )

            # for the volume, it is easy, the volume backscatter is convolved with the pfs
            pfs_backscatter_volume = scipy.signal.convolve(pfs, backscatter_volume, mode="full")

            # for the surface, it depends on the incidence angle, first interpolate in time
            # and then multiply by the pfs (no convolution)
            def interpolate_backscatter(backscatter):
                # interpolate in time + shift the backscatter to the nominal gate
                return np.interp(
                    extended_t_gate - self.sensor.nominal_gate / self.sensor.pulse_bandwidth, t_inc_sample, backscatter, left=0
                )

            pfs_backscatter_surface = np.zeros_like(pfs_backscatter_volume)
            pfs_backscatter_surface[0 : len(pfs)] = interpolate_backscatter(backscatter_surface) * pfs

            pfs_backscatter_interfaces = np.zeros_like(pfs_backscatter_volume)

            # for the interface, this is the same as the surface, except it must be shifted in time (basic convolution)
            for i in range(backscatter_interfaces.shape[1]):
                # this is a bit hacky, we should rather directly compute and use the backscatter per interface instead
                # of the subgate interface backscatter
                if backscatter_interfaces[0, i] > 0:
                    pfs_backscatter_interfaces[i : i + len(pfs)] += (
                        interpolate_backscatter(backscatter_interfaces[:, i]) * pfs
                    )

            # now take into account the PTR + PDF assuming both are gaussian
            if (self.sensor.pulse_sigma > 0) or (sigma_surface > 0):
                # perform the last convolution
                def do_convolve_by_PTR_PDF(backscatter):
                    # perform the convolution and cut the first part due to t_gate symmetrization (ensure the nominal
                    # gate is at the right place)
                    return scipy.signal.convolve(ptr_pdf, backscatter, mode="full")[len(sym_t_gate) // 2:]

                waveform_surface = do_convolve_by_PTR_PDF(pfs_backscatter_surface)
                waveform_volume = do_convolve_by_PTR_PDF(pfs_backscatter_volume)
                waveform_interface = do_convolve_by_PTR_PDF(pfs_backscatter_interfaces)

            else:
                # no need to add PTR and PDF
                ptr_pdf = 1 / self.sensor.pulse_bandwidth

                waveform_surface = pfs_backscatter_surface * ptr_pdf
                waveform_volume = pfs_backscatter_volume * ptr_pdf
                waveform_interface = pfs_backscatter_interfaces * ptr_pdf

            if self.return_contributions:
                return np.vstack((waveform_surface, waveform_interface, waveform_volume))
            else:
                return waveform_surface + waveform_interface + waveform_volume

    def gate_depth(self, eps=None):
        # """return gate depth that cover the snowpack for a regular time sampling"""

        if eps is None:
            eps = [em.effective_permittivity().real for em in self.emmodels]

        c_lay = C_SPEED / np.sqrt(eps).real
        t_lay = 2 * np.cumsum(self.snowpack.layer_thicknesses / c_lay)
        t_lay = np.insert(t_lay, 0, 0)

        # regular sampling in time to cover the whole snowpack
        ngate = max(np.ceil(t_lay[-1] * (self.sensor.pulse_bandwidth * self.oversampling)), 1)
        t_gate = np.arange(0, ngate + 1) / (self.sensor.pulse_bandwidth * self.oversampling)
        # position of the gates in the snow, accounting for the wave speed.
        z_gate = np.interp(t_gate, t_lay, self.snowpack.z)

        z_gate[-1] += 0.01 * (
            z_gate[-1] - z_gate[-2]
        )  # slightly increase the last gate to guarantee that it is after the substrate
        return z_gate, t_gate

    def combined_depth_grid(self):
        z_lay = self.snowpack.z

        # merge both depth array (layer and gate)
        z = np.concatenate((z_lay, self.z_gate))
        i = np.argsort(z)
        z = z[i]

        # bool array where interfaces, gates and layers are
        b_interface = np.concatenate((np.ones_like(z_lay, dtype=bool), np.zeros_like(self.z_gate, dtype=bool)))[i]
        b_gate = ~b_interface

        b_layer = b_interface.copy()
        b_layer[i == len(z_lay) - 1] = False  # remove the last interface

        dz = np.diff(z)  # subgate thickness
        return z[:-1], dz, b_gate, b_layer[:-1], b_interface

    def vertical_scattering_distribution(self, return_contributions, mu_i=1.0):
        # """Compute the vertical backscattering distribution due to "grain" or volume scattering (symbol pvg in Eq 9 in Lacroix 2008) and
        # "interfaces" or 'surface' scattering (symbol pvl in Eq 9 in Lacroix 2008)

        # :param mu: cosine of the incidence angles. Only the dependence on the surface scattering depend on mu_i

        # Returns:
        #     ndarray: Backscattering distribution.
        # """
        mu_i = np.atleast_1d(mu_i)

        # propagation time
        eps = np.array([em.effective_permittivity().real for em in self.emmodels])

        # compute the merged depth grid including gates and layers
        z_top, dz, b_gate, b_layer, b_interface = self.combined_depth_grid()

        # layer extinction
        layer_extinction = [np.mean(em.ke(mu=[1.0]).diagonal) for em in self.emmodels]

        subgate_layer_extinction = fill_forward(layer_extinction, b_layer)

        # layer backscatter
        # backward scattering (take VV, is equal to HH) # nadir backward scattering. a.k.a gamma in Matzler's notation. We neglect mu_i.
        backward_scattering = np.array(
            [
                em.phase(mu_s=-1.0, mu_i=1.0, dphi=np.pi, npol=2)[0, 0].squeeze().real / (4 * np.pi)
                for em in self.emmodels
            ]
        )  # 4pi normalisation coming from Eq. 2 in Picard et al. 2018

        backward_scattering /= eps.real  # take into account the divergence of the upwelling stream due to refraction
        backward_scattering = fill_forward(backward_scattering, b_layer)

        subgate_dtau = 2 * subgate_layer_extinction * dz  # two-way optical depth of the layer
        # analytical integration of the backscatter in the subgate grid (where extinction is constant)
        subgate_backscatter_v = (1 - np.exp(-subgate_dtau)) / (2 * subgate_layer_extinction) * backward_scattering

        # now compute total attenuation
        # layer 'volume' attenuation: interpolate to z
        subgate_tau_v = np.insert(np.cumsum(subgate_dtau), 0, 0)  # two-way
        subgate_attenuation_v = np.exp(
            -subgate_tau_v
        )  # np.inf allow to set attenuation to 0 for gates > snowdepth or gates < 0

        # 'interface' attenuation due to transmission. We neglect mu_i.
        transmission = [
            i.coherent_transmission_matrix(self.sensor.frequency, eps_1, eps_2, mu1=1.0, npol=2)[0, 0]
            for i, eps_1, eps_2 in zip(self.snowpack.interfaces, np.insert(eps[:-1], 0, 1), eps)
        ]
        cum_transmission = np.cumprod(np.array(transmission) ** 2, axis=0)  # two-way transmission

        subgate_attenuation_i = np.insert(fill_forward(cum_transmission, b_layer), 0, 1.0)

        # attenuation at the top of a layer (below the interface) is the product of layer and interface attenuation
        subgate_backscatter_v *= subgate_attenuation_v[:-1] * subgate_attenuation_i[1:]

        # compute the interface backscatter. This can depend on theta
        # for each gate (PFS convolution).

        eps_upper_interface = np.insert(eps[:-1], 0, 1.0)
        mu_upper_interface = np.sqrt(1 - (1 - mu_i[None, :]) / eps_upper_interface[:, None]).real

        if self.compute_coherent_reflection:
            flat = Flat()

        # TODO: add the coherent component
        interface_echo = [
            (
                i.diffuse_reflection_matrix(self.sensor.frequency, eps_1, eps_2, mu_s=mu, mu_i=mu, dphi=np.pi, npol=2)
                .diagonal[0]
                .squeeze()
                / eps_1.real
            )
            + (
                flat.specular_reflection_matrix(self.sensor.frequency, eps_1, eps_2, mu1=mu, npol=2)
                .diagonal[0]
                .squeeze()
                # use the rms_height of the interface
                * coherent_reflection_factor(self.sensor, i.roughness_rms, mu)
                if (self.compute_coherent_reflection) and hasattr(i, "roughness_rms")
                else 0
            )
            for i, eps_1, eps_2, mu in zip(self.snowpack.interfaces, eps_upper_interface, eps, mu_upper_interface)
        ]

        # note that the division by eps_1 takes into account the divergence of the upwelling stream due to refraction

        if self.snowpack.substrate is not None:
            mu_last_layer = np.sqrt(1 - (1 - mu_i) / eps[-1]).real
            # TODO: add the coherent component
            interface_echo += [
                (
                    self.snowpack.substrate.diffuse_reflection_matrix(
                        self.sensor.frequency, eps[-1], mu_s=mu_last_layer, mu_i=mu_last_layer, dphi=np.pi, npol=2
                    )
                    .diagonal[0]
                    .squeeze()
                    / eps[-1].real
                )
                + (
                    flat.specular_reflection_matrix(
                        self.sensor.frequency,
                        eps[-1],
                        self.snowpack.substrate.permittivity(self.sensor.frequency),
                        mu1=mu_last_layer,
                        npol=2,
                    )
                    .diagonal[0]
                    .squeeze()
                    # use the rms_height of the interface and if not use the "macroscopic" sigma_surface
                    * coherent_reflection_factor(self.sensor, self.snowpack.substrate.roughness_rms, mu_last_layer)
                    if self.compute_coherent_reflection
                    else 0
                )
            ]
        else:
            # no echo from the bottom
            interface_echo += [np.zeros_like(interface_echo[-1])]

        if len(mu_upper_interface[0]) > 1:
            # convert the scalar into matrix to get an homogeneous list before transformation to numpy array
            interface_echo = [np.full(len(mu_upper_interface[0]), m) if np.ndim(m) == 0 else m for m in interface_echo]

        interface_echo = np.transpose(interface_echo)

        if len(mu_i) > 1:
            # check that the first dimension is that of mu_i
            assert interface_echo.shape[0] == len(mu_i)

        # attenuation at the bottom of the layer is the product of layer and interface attenuation
        subgate_backscatter_i = fill(interface_echo, b_interface) * subgate_attenuation_v * subgate_attenuation_i

        # compute the primitive of the subgate backscatter, select the gate interval and differentitate to get the integrated backscatter
        # over each gate
        if self.return_contributions or (self.theta_inc_sampling > 1):
            # volume contribution
            subgate_backscatter_v = np.insert(subgate_backscatter_v, 0, 0)
            gate_backscatter_v = np.diff(np.insert(np.cumsum(subgate_backscatter_v)[b_gate], 0, 0))

            # save the surface contribution
            subgate_backscatter_s = subgate_backscatter_i[..., 0].copy()
            subgate_backscatter_i[..., 0] = 0
            gate_backscatter_i = np.diff(
                np.insert(np.cumsum(subgate_backscatter_i, axis=-1)[..., b_gate], 0, 0, axis=-1), axis=-1
            )

            gate_backscatter_s = np.zeros_like(gate_backscatter_i)
            gate_backscatter_s[..., 0] = subgate_backscatter_s

            return np.vstack((gate_backscatter_s, gate_backscatter_i, gate_backscatter_v))
        else:
            assert len(mu_i) == 1
            subgate_backscatter = subgate_backscatter_i + np.insert(subgate_backscatter_v, 0, 0)

            gate_backscatter = np.diff(np.insert(np.cumsum(subgate_backscatter)[b_gate], 0, 0))
            return gate_backscatter

    def PFS_numerical(self, tau):
        # Compute the PFS using numerical integration of the phi axis
        # the sensor must define the antenna gain function which take theta and phi as arguments.

        h = self.sensor.altitude

        def integrand(phi, theta):
            return self.waveform_model.G(theta, phi) ** 2  # G^2

        e = C_SPEED / h * tau

        rho_h = np.sqrt((e / 2 + 1) ** 2 - 1)
        theta = np.arctan(rho_h)

        integral = np.empty_like(tau)
        for i, ttau in enumerate(tau):
            integral[i], err = scipy.integrate.quad(integrand, 0, 2 * np.pi, agrs=theta[i], epsrel=1e-4)

        return (
            self.sensor.wavelength**2
            * C_SPEED
            / (2 * (4 * np.pi) ** 3 * self.sensor.altitude**3)
            * integral
            * np.sqrt(1 + rho_h**2)
        )


def fill_forward(a, where, axis=-1):
    # Create an array of size where, which is filled with a and fill foward from the beginning of where to the end.
    assert np.array(a).dtype == np.float64
    idx = np.cumsum(where)
    return np.take(np.insert(np.array(a, dtype=np.float64), 0, np.nan, axis=-1), idx, axis=-1)


def fill(a, where, novalue=0):
    # Fill along last axis
    a = np.array(a)
    out = np.full(a.shape[:-1] + (where.shape[0],), 0.0)

    assert np.sum(where) == a.shape[-1]
    np.place(out, np.broadcast_to(where, out.shape), a)  # where is broadcast onto the last dim
    return out


def coherent_reflection_square_decay(sensor):
    # for the sqrt(2),see Claude de Rijke-Thomas thesis, p 106-111.
    beta0 = np.sqrt(C_SPEED / (sensor.pulse_bandwidth * sensor.altitude)) * np.sqrt(2)

    beta12 = 1 / (sensor.wavenumber * sensor.altitude * beta0) ** 2 + beta0**2 / 4
    return beta12


def coherent_reflection_factor(sensor, roughness_rms, mu):
    """return the factor to account for the coherent echo due to the spherical wave (see Fung and Eom, 1983) equation 6.
    This neglects the macroscopic slope of the terrain, which should be included in principle.
    """

    sintheta2 = 1 - mu**2  # we neglect the slope
    theta2 = sintheta2  # approximation

    beta12 = coherent_reflection_square_decay(sensor)

    return np.exp(-4 * (sensor.wavenumber * roughness_rms) ** 2 - theta2 / beta12) / beta12 / (4 * np.pi)
