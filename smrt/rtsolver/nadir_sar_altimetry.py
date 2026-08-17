"""This module provides Nadir SAR Mode Altimetry rtsolver.

This rtsolver computes delay Doppler maps and waveforms for the given snowpack and sensor (or complex terrain soon)
according to a delay doppler model specified as input (and available in the directory
smrt/rtsolver/delay_doppler_model). As opposed to other common rtsolvers in SMRT that behave independently, this solver
delegates part of the computation to the selected delay doppler model. The results and the capabilities highly depend on
the selected delay doppler model. For instance some delay doppler models can handle satellite mis-pointing, asymmetric
antenna and/or Digital Elevation Model. Some models can compute delay Doppler maps without the slant range correction.
Moreover they differ in many computational aspects that need to be understood before selecting one or another.

More precisely this rtsolver first computes the first-order backscatter coming for each "level", including the surface,
each interface, and the volume split in subgate intervals. In a second step it leverages the delay doppler model to
compute how each level contributes to the delay Doppler map. It then sums all these contributions to get the final delay
Doppler map and are returned as the AltimeterResult object. Ultimately, the waveform (= the sum in the Doppler
dimension) can be obtained from this object. It is also possible to get the contributions from the surface, interfaces
and volume independently (that is three different delay Doppler maps).

.. note:
    With this RT solver, if using Geometrical Optics for rough surface/interface modeling, it is strongly advised to use
    :py:mod:`~smrt.interface.geometrical_optics_backscatter` instead of :py:mod:`~smrt.interface.geometrical_optics` for
    the reasons explained in the documentation of those modules.

Main approximations:
    - Backscatter is computed assuming only first order scattering. The propagation is then simply calculated with the
      extinction.
    - Small-angle approximation: to compute delay, the paths in the snow are assumed vertical. We neglect the 1/cos(theta)
      lengthening of the path

Usage:
    Basic usage with default settings and IBA EM model and Dinardo18 DDM model:
        >>> m = make_model("iba", "nadir_sarm_altimetry", rtsolver_options = dict(delay_doppler_model="dinardo18"))

    or using make_rtsolver:
        >>> m = make_model("iba", make_rtsolver("nadir_sarm_altimetry",
                                                delay_doppler_model="dinardo18"))

    To specifiy DDM settings, for instance to use Ray15 DDM model with specific PTRs:
        >>> ddm_options = dict(ptr_time="gaussian", ptr_doppler="gaussian-smrt", slant_range_correction=True)
        >>> m = make_model("iba", "nadir_sarm_altimetry",
                            rtsolver_options = dict(delay_doppler_model="ray15",
                                                    delay_doppler_model_options=ddm_options))

    An alternative using make_rtsolver:
        >>> ddm_options = dict(ptr_time="gaussian", ptr_doppler="gaussian-smrt", slant_range_correction=True)
        >>> m = make_model("iba", make_rtsolver("nadir_sarm_altimetry",
                                                delay_doppler_model=delay_doppler_model,
                                                delay_doppler_model_options=ddm_options))

References:
    - Picard, G., Murfit, J., Zakharova, E., Zeiger, P., Arnaud, L., Aublanc, J., Landy, J., Scagliola, M., and Duguay,
      C. (2025). Simulating SAR altimeter echoes from cryospheric surfaces with the Snow Microwave Radiative Transfer
      (SMRT) model version sarm-v0. Geosci. Model Dev. Discuss., submitted.
    - Larue, F., Picard, G., Aublanc, J., Arnaud, L., Robledano-Perez, A., Meur, E. L., Favier, V., Jourdain, B.,
      Savarino, J., & Thibaut, P. (2021). Radar altimeter waveform simulations in Antarctica with the Snow Microwave
      Radiative Transfer Model (SMRT). Remote Sensing of Environment, 263, 112534.
      https://doi.org/10.1016/j.rse.2021.112534

"""

from typing import Type

import numpy as np
import numpy.typing as npt
import scipy.signal
import xarray as xr

from smrt.core import lib
from smrt.core.error import SMRTError, smrt_warn
from smrt.core.globalconstants import C_SPEED
from smrt.core.plugin import import_class
from smrt.core.result import AltimetryResult
from smrt.interface.flat import Flat

from .delay_doppler_model.delay_doppler_utils import delay_sampling_vector, doppler_frequency_vector
from .delay_doppler_model.dinardo18 import Dinardo18


class NadirSARAltimetry(object):
    """RTSolver to calculate Delay Doppler Maps and waveforms in the SAR mode (InSAR and fully focused are not
       implemented yet) by including backscatter of the surface, interfaces and volume.

    :param delay_doppler_model: model (string or class) to use to compute the delay_doppler_map for an interface.
    :param delay_doppler_model_options: options to apply to instantiate the delay_doppler_map model. The available
        options may be different for each model. Refer to the documentation of each model. The options
        slant_range_correction, ptr_time and ptr_doppler are often available.
    :param oversampling_time: integer number defining the number of subgates used for the computation in each altimeter
        gate. This is equivalent to multiply the bandwidth by this number. It is used/necessary to perform more accurate
        computation.
    :param oversampling_doppler: integer number defining the number of sub-Doppler beams used for the computation. This
        is equivalent to multiply the bandwidth by this number. It is used/necessary to perform more accurate
        computation.
    :param return_oversampled: by default the backscatter is returned for each gate. If set to True, the oversampled
        waveform is returned instead. See the 'oversampling' argument.
    :param return_contributions: controls the returned contributions as follows: - False or "no": total backscatter
        only. - True or "basic": total, volume, surface and interface backscatter. - "basic coherent": total, volume,
        incoherent surface, coherent surface, incoherent interface, and coherent interface backscatter contributions. -
        "full": total, volume, and all interfaces separated. - "full coherent": total, volume, and all interface
        separated split into the coherent and incoherent components.
    :param skip_convolutions: return the vertical backscatter without the convolution by the PFS, PTR_time, and
        PTR_doppler, if set to True.
    :param theta_inc_sampling: number of subdivisions used to calculate the incidence angular variations of surface and
        interface backscatter (the higher the better but the more computationally expensive).
    :param error_handling: If set to "exception" (the default), raise an exception in case of error, stopping the code.
        If set to "nan", return a nan, so the calculation can continue, but the result is of course unusable and the
        error message is not accessible. This is only recommended for long simulations that sometimes produce an error.
    """

    # this specifies which dimension this solver is able to deal with. Those not in this list must be managed by the caller (Model object)
    # e.g. here, frequency, time, ... are not managed
    _broadcast_capability = {}  # "theta_inc", "polarization_inc", "theta", "phi", "polarization"}

    def __init__(
        self,
        delay_doppler_model=None,
        delay_doppler_model_options={},
        oversampling_time=4,
        oversampling_doppler=4,
        return_oversampled=False,
        skip_convolutions=False,
        return_contributions=False,
        theta_inc_sampling=8,
        error_handling="exception",
        prune_deep_snowpack=True,
    ):
        # """

        # """

        super().__init__()

        if delay_doppler_model is None:
            self.delay_doppler_model_class = Dinardo18
        elif isinstance(delay_doppler_model, str):
            self.delay_doppler_model_class = import_class("rtsolver.delay_doppler_model", delay_doppler_model)
        elif isinstance(delay_doppler_model, type):
            self.delay_doppler_model_class = delay_doppler_model
        else:
            raise SMRTError(f"Incorrect type for the delay_doppler_model parameter: {type(delay_doppler_model)}.")

        self.delay_doppler_model_options = delay_doppler_model_options

        print("delay doppler model:", self.delay_doppler_model_class)
        self.error_handling = error_handling
        self.oversampling_time = oversampling_time
        self.oversampling_doppler = oversampling_doppler

        # deal with the return contributions
        if return_contributions is True:
            return_contributions = "basic"
        if not return_contributions:
            return_contributions = "no"
        self.return_coherent = return_contributions.endswith(" coherent")
        self.return_contributions = return_contributions.replace(" coherent", "")

        if self.return_contributions not in [None, False, "no", "basic", "full"]:
            raise SMRTError(
                'Invalid return_contributions. It must None, "no", basic", "basic coherent", "full" or  "full coherent"'
            )

        self.return_oversampled = return_oversampled
        self.skip_convolutions = skip_convolutions
        self.theta_inc_sampling = theta_inc_sampling

        self.prune_deep_snowpack = prune_deep_snowpack

    def solve(self, snowpack_or_terrain, emmodels, sensor, atmosphere=None):
        """Solve the radiative transfer equation for a given snowpack (or terrain with multiple snowpack), emmodels and sensor configuration."""
        if sensor.theta_inc != 0:
            raise SMRTError(
                "This solver is for nadir looking altimeter only. Satellite mis-pointing (pitch and roll) is allowed for some waveform model."
            )
        assert atmosphere is None

        self.terrain_info = snowpack_or_terrain.terrain_info
        self.sensor = sensor  # setting this make this solver incompatible with // computing

        # check that all interfaces have roughness_rms
        if any(not hasattr(interface, "roughness_rms") for interface in snowpack_or_terrain.interfaces):
            raise SMRTError(
                "nadir_sarm_altimetry only works with interfaces that define 'roughness_rms' to compute the coherent reflection. "
            )

        if (snowpack_or_terrain.substrate is not None) and not hasattr(snowpack_or_terrain.substrate, "roughness_rms"):
            raise SMRTError(
                "nadir_sarm_altimetry only works with substrates that define 'roughness_rms' to compute the coherent reflection. "
            )

        # prune the snowpack if this option is set
        if self.prune_deep_snowpack:
            # max depth seen by the sensor in vacuum
            max_visible_depth = sensor.ngate / sensor.pulse_bandwidth * C_SPEED / 2
            # bottom depths of the layers
            depths = snowpack_or_terrain.bottom_layer_depths
            if depths[-1] > max_visible_depth:  # the last layer is below the maximum reach of the sensor
                nmax = np.searchsorted(depths, max_visible_depth)
                # print(f"Prune: {nmax} / {snowpack_or_terrain.nlayer}")
                self.snowpack = snowpack_or_terrain.shallow_copy(cut_bottom=nmax + 1)
                self.emmodels = emmodels[: nmax + 1]
            else:
                self.snowpack = snowpack_or_terrain
                self.emmodels = emmodels
        else:
            self.snowpack = snowpack_or_terrain  # terrain is not used right now
            self.emmodels = emmodels

        self.interfaces = self.snowpack.interfaces
        # gather all interfaces
        self.all_interfaces = self.interfaces
        if self.snowpack.substrate is not None:
            self.all_interfaces = self.all_interfaces + [self.snowpack.substrate]  # copy and add

        # compute the effective permittivity
        self.effective_permittivity = np.array([em.effective_permittivity() for em in self.emmodels])

        # create the DDM model
        ddm_options = self.delay_doppler_model_options | dict(
            oversampling_time=self.oversampling_time, oversampling_doppler=self.oversampling_doppler
        )
        self.delay_doppler_model = self.delay_doppler_model_class(sensor, **ddm_options)

        if hasattr(self.delay_doppler_model, "delay_doppler_map_with_sigma0func"):
            # we need to calculate the interface backscatter at several incidence angles.
            # Let choose something optimal to cover the footprint
            t_inc_sample = np.linspace(0, self.sensor.ngate / self.sensor.pulse_bandwidth, self.theta_inc_sampling + 1)
            # cosine is adjacent divided by hypotenuse. This neglects the Earth's sphericity.
            mu_i = 1.0 / (1.0 + C_SPEED * t_inc_sample / sensor.altitude)
        else:
            # for models based on constant sigma0 and geometrical optics, we only need strict nadir
            mu_i = 1

        local_mu_i = local_incidence_cosine(sensor, mu_i)

        # compute the vertical backscatter
        self.z_gate, _, self.t_interface = self.gate_depth()
        gate_backscatter_v, backscatter_i = self.vertical_scattering_distribution(local_mu_i)

        N = self.sensor.ngate * self.oversampling_time
        # need padding ? if yes, pad the end with zeros
        if gate_backscatter_v.shape[-1] < N:
            gate_backscatter_v = np.append(
                gate_backscatter_v,
                np.zeros(gate_backscatter_v.shape[:-1] + (N - gate_backscatter_v.shape[-1],)),
                axis=-1,
            )
        # TODO: deal with self.skip_convolutions
        if self.skip_convolutions:
            raise NotImplementedError("skip_convolutions is not implemented yet")

        # combine the vertical backscatter with the delay doppler maps
        ddm, contributions = self.combine_backscatter_and_delay_doppler_map(
            gate_backscatter_v, backscatter_i, local_mu_i
        )

        #
        # prepare the output in the Result object
        #

        t_gate = delay_sampling_vector(sensor, self.oversampling_time)
        fdoppler = doppler_frequency_vector(sensor, self.oversampling_doppler)

        # add the theta_inc and theta dimension. They are required by Results to work well
        theta_inc_deg = [0]
        coords = [
            ("delay", t_gate),
            ("doppler_frequency", fdoppler),
            ("theta_inc", theta_inc_deg),
            ("theta", theta_inc_deg),
        ]
        ddm = ddm[..., np.newaxis, np.newaxis]

        if contributions:
            coords = [("contribution", contributions)] + coords

        ddm = xr.DataArray(ddm, coords=coords)

        # add useful coordinates
        ddm = ddm.assign_coords(
            gate=ddm.delay * sensor.pulse_bandwidth + sensor.nominal_gate,
            doppler_bin=ddm.doppler_frequency / sensor.pulse_repetition_frequency * sensor.ndoppler,
        )

        # downsample if necessary
        if not self.return_oversampled and ((self.oversampling_time > 1) or (self.oversampling_doppler > 1)):
            t_gate = t_gate[:: self.oversampling_time]
            self.z_gate = self.z_gate[:: self.oversampling_time]
            ddm = ddm.coarsen(
                dim=dict(delay=self.oversampling_time, doppler_frequency=self.oversampling_doppler)
            ).mean()

        # create the result object
        res = AltimetryResult(ddm)

        # add some extra information
        if len(self.z_gate) >= len(t_gate):
            # shorten
            self.z_gate = self.z_gate[0 : len(t_gate)]
        else:
            # extend with nan's
            self.z_gate = np.append(self.z_gate, np.full(len(t_gate) - len(self.z_gate), np.nan))

        res.z_gate = xr.DataArray(
            self.z_gate, coords=[("delay", t_gate + self.sensor.nominal_gate / self.sensor.pulse_bandwidth)]
        )
        return res

    def combine_backscatter_and_delay_doppler_map(self, backscatter_v, backscatter_i, mu_i):
        # calculate the delay doppler maps for the volume and the interfaces and multiply/convolve with the backscatter.
        # For the interfaces, there are three very different cases to consider:
        # 1) sigma0 is constant with the incidence angle -> easy, we only need to compute a single DDM.
        # 2) sigma0 is given by the geometrical optics approximation and the model is built on this -> the calculation
        # as a function of the incidence angle can be vectorized, and the code avoid recalculation when different
        # interfaces have the same mean_square_slope
        # 3) the model can deal with any sigma0 function.

        def convolve_ddm(ddm, backscatter):
            # perform the time convolution between the ddm and the volume backscatter signal
            N = ddm.shape[0]
            return scipy.signal.convolve(ddm, backscatter[:N, np.newaxis], mode="full")[:N, :]

        def shift_ddm(ddm, time):
            # interface return a dirac echo in time, the convolution is a simple shift
            itime = int(np.round(time * self.sensor.pulse_bandwidth * self.oversampling_time))
            if itime == 0:
                return ddm
            assert 0 <= itime < ddm.shape[0]

            ddm_o = np.zeros_like(ddm)
            ddm_o[itime:, :] = ddm[:-itime, :]
            return ddm_o

        # ddm for all the interfaces. Replicate a zero ddm to start with
        zero = np.zeros((self.sensor.ngate * self.oversampling_time, self.sensor.ndoppler * self.oversampling_doppler))
        ddm_i = [zero] * len(self.all_interfaces)
        coh_ddm_i = [zero] * len(self.all_interfaces)

        # separate the coherent and incoherent contributions
        coh_backscatter_i = backscatter_i[..., 0, :]
        backscatter_i = backscatter_i[..., 1, :]

        #
        # Geometrical optics model
        #
        if hasattr(self.delay_doppler_model, "delay_doppler_map_with_GO"):
            # the model uses geometrical optics for the interfaces. Let's compute the response for each different value of MSS

            # prepare a set of unique mean_square_slope values in the interfaces that contributes to the backscatter
            # First for the volume: np.inf means no dependency to the incidence angle
            mean_square_slope = set([np.inf])

            # Second for the coherent backscatter
            if np.any(coh_backscatter_i > 0):
                # this is the coherent reflection which always decays the same way, depending only on sensor characteristics
                coherent_decay = coherent_reflection_square_decay(self.sensor)
                mean_square_slope.add(coherent_decay)

            # Last for each interface
            for i, interface in enumerate(self.all_interfaces):
                if backscatter_i[i] > 0:
                    try:
                        mean_square_slope.add(interface.mean_square_slope)
                    except AttributeError:
                        raise SMRTError(
                            f'The delay_doppler_model "{self.delay_doppler_model}" relies on geometrical optics interfaces only (or non scattering interfaces).'
                        )

            # gather all the mean_square_slope
            mean_square_slope = list(mean_square_slope)

            # calculate the DDMs in one call for all mean_square_slopes
            ddm_list = self.delay_doppler_model.delay_doppler_map_with_GO(
                terrain_info=self.terrain_info, mean_square_slope=mean_square_slope
            )
            ddm_dict = dict(zip(mean_square_slope, list(ddm_list)))  # transform the list into a dict

            # scale the backscatter with the delay Doppler maps
            # first the volume
            ddm_volume = convolve_ddm(ddm_dict[np.inf], backscatter_v)

            # second the interfaces
            for i, interface in enumerate(self.all_interfaces):
                if backscatter_i[i] > 0:
                    ddm_i[i] = shift_ddm(ddm_dict[interface.mean_square_slope], self.t_interface[i]) * backscatter_i[i]
                if coh_backscatter_i[i] > 0:
                    coh_ddm_i[i] = shift_ddm(ddm_dict[coherent_decay], self.t_interface[i]) * coh_backscatter_i[i]
        #
        # sigma0func
        #
        elif hasattr(self.delay_doppler_model, "delay_doppler_map_with_sigma0func"):
            # the model uses a free function for the surface and interface backscatter.

            # actions hold for each component and each interface what must be done:
            # either None=nothing
            # either use a sigma0 function
            # or a multiplicative factor
            actions = []

            for backscatt in (backscatter_i, coh_backscatter_i):  # iterate over the incoherent and coherent components
                for i, interface in enumerate(self.all_interfaces):  # iterate over the interfaces
                    if np.all(backscatt[:, i] == 0):
                        # no backscatter... skip the complex calculations of the DDM
                        actions.append((None, None))
                    elif np.all(backscatt[:, i] == backscatt[0, i]):  # all backscatter are equal ?
                        # no need to use a function for sigma0. The DDM for the volume can be used
                        actions.append(("mul", backscatt[0, i]))
                    else:
                        # need to define a function able to compute the backscatter for any theta. We use interpolation
                        # i.e. mu = np.cos(theta)
                        def sigma0(mu: npt.NDArray, bs=backscatt[:, i]) -> npt.NDArray:
                            # it is important to add the bs= argument, otherwise the last backscatt[:, i] is taken due
                            # to the local context
                            # we know that mu_i is decreasing, so the minus
                            backscatter_map = np.interp(-mu, -mu_i, bs)
                            return backscatter_map

                        actions.append(("func", sigma0))

            # compute the ddm for all sigma0
            ddm_list = self.delay_doppler_model.delay_doppler_map_with_sigma0func(  # type: ignore
                terrain_info=self.terrain_info, sigma0=[1] + [func for action, func in actions if action == "func"]
            )  # [1] is for the volume and the surface with constant backscatter. It is very important!

            # convolve the ddm with the volume backscatter
            ddm_volume = convolve_ddm(ddm_list[0], backscatter_v)

            # for each interface calculate the ddm
            j = 1  # index in the ddm_list (start at the first calculated interface)

            # merge the ddms (first incoherent, second coherent)
            all_ddm_i = ddm_i + coh_ddm_i
            for i, (action, param) in enumerate(actions):  # loop over the actions in the same order
                if action == "func":  # a specific DDM has been calculated for this interface
                    all_ddm_i[i] = ddm_list[j]
                    j += 1  # increment the index of calculated function
                elif action == "mul":  # all backscatter are equal
                    # use the DDM for the volume (without dependency can be used here) and the mulfactor
                    all_ddm_i[i] = param * ddm_list[0]

            # split the incoherent and coherent
            ddm_i, coh_ddm_i = all_ddm_i[: len(ddm_i)], all_ddm_i[len(ddm_i) :]

        else:
            # sigma0 must be independent of the incidence angle
            # compute a single ddm_v
            ddm_v = self.delay_doppler_model.delay_doppler_map(terrain_info=self.terrain_info)

            # TODO: could be optimized when return_contributions is False, but I'm not sure this represent a significant gain

            # first convolve with the volume
            ddm_volume = convolve_ddm(ddm_v, backscatter_v)

            if np.any(coh_backscatter_i > 0):
                smrt_warn("This delay Doppler model is not compatible with coherent backscatter")
            # second convolve with the interfaces
            for i, interface in enumerate(self.all_interfaces):
                if (backscatter_i[i] > 0) or (coh_backscatter_i[i] > 0):
                    shift = shift_ddm(ddm_v, self.t_interface[i])
                    if backscatter_i[i] > 0:
                        ddm_i[i] = shift * backscatter_i[i]
                    if coh_backscatter_i[i] > 0:
                        coh_ddm_i[i] = shift * coh_backscatter_i[i]

        # split the contributions
        ddm_surface = ddm_i[0] + coh_ddm_i[0]

        coh_ddm_all_internal_interfaces = sum(coh_ddm_i[1:])
        ddm_all_internal_interfaces = sum(ddm_i[1:])

        # compute the total ddm
        ddm_total = ddm_surface + ddm_all_internal_interfaces + coh_ddm_all_internal_interfaces + ddm_volume

        def cm(m):  # condition the matrix shape
            return np.full_like(ddm_total, m) if np.isscalar(m) else m

        # return the contributions or the total only
        if self.return_contributions == "basic":
            if self.return_coherent:
                contributions = [
                    "incoherent surface",
                    "coherent surface",
                    "incoherent interfaces",
                    "coherent interfaces",
                    "volume",
                    "total",
                ]
                m = np.stack(
                    (
                        ddm_i[0],  # surface
                        coh_ddm_i[0],  # surface coherent
                        cm(ddm_all_internal_interfaces),  # interfaces
                        cm(coh_ddm_all_internal_interfaces),  # interfaces coherent
                        ddm_volume,
                        ddm_total,
                    )
                )
                return m, contributions
            else:
                contributions = ["surface", "interfaces", "volume", "total"]
                return np.stack(
                    (
                        ddm_surface,  # surface
                        cm(ddm_all_internal_interfaces + coh_ddm_all_internal_interfaces),  # interfaces
                        ddm_volume,
                        ddm_total,
                    )
                ), contributions

        elif self.return_contributions == "full":
            if self.return_coherent:
                contributions = (
                    [f"incoherent interface {i}" for i in range(len(ddm_i))]
                    + [f"coherent interface {i}" for i in range(len(ddm_i))]
                    + ["volume", "total"]
                )
                return np.concatenate(
                    (
                        ddm_i,  # all interfaces
                        coh_ddm_i,  # all interfaces coherent
                        ddm_volume[np.newaxis, ...],
                        ddm_total[np.newaxis, ...],
                    )
                ), contributions
            else:
                contributions = [f"interface {i}" for i in range(len(ddm_i))] + ["volume", "total"]
                return np.concatenate((ddm_i, ddm_volume[np.newaxis, ...], ddm_total[np.newaxis, ...])), contributions
        else:
            return ddm_total, []

    def gate_depth(self):
        """Return gate depths that cover the snowpack for regular time sampling."""
        c_layer = C_SPEED / np.sqrt(self.effective_permittivity).real
        t_layer = 2 * np.cumsum(self.snowpack.layer_thicknesses / c_layer)
        t_interface = np.insert(t_layer, 0, 0)

        # regular sampling in time to cover the whole snowpack
        ngate = max(np.ceil(t_interface[-1] * (self.sensor.pulse_bandwidth * self.oversampling_time)), 1)
        t_gate = np.arange(0, ngate + 1) / (self.sensor.pulse_bandwidth * self.oversampling_time)
        # position of the gates in the snow, accounting for the wave speed.
        z_gate = np.interp(t_gate, t_interface, self.snowpack.z)

        # guarantee that the first gate is within the snowpack
        z_gate[0] = 1e-25

        z_gate[-1] += 0.01 * (
            z_gate[-1] - z_gate[-2]
        )  # slightly increase the last gate to guarantee that it is after the substrate
        return z_gate, t_gate, t_interface

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

    def vertical_scattering_distribution(self, mu_i):
        """Compute the vertical backscattering distribution due to "grain" or volume scattering, "interfaces" or 'surface' scattering"""
        mu_i = np.atleast_1d(mu_i)

        # compute the merged depth grid including gates and layers
        z_top, dz, b_gate, b_layer, b_interface = self.combined_depth_grid()

        # compute layer extinction
        layer_extinction = [np.mean(em.ke(mu=[1.0]).diagonal) for em in self.emmodels]
        # ... and subsample to the combined grid
        subgate_layer_extinction = _fill_forward(layer_extinction, b_layer)

        ############################################################################################
        # compute layer volume backscatter
        # backward scattering (take VV, is equal to HH) # nadir backward scattering. a.k.a gamma in Matzler's notation. We neglect mu_i.
        backward_scattering = np.array(
            [
                em.phase(mu_s=-1.0, mu_i=1.0, dphi=np.pi, npol=2)[0, 0].squeeze().real / (4 * np.pi)
                for em in self.emmodels
            ]
        )  # 4pi normalisation coming from Eq. 2 in Picard et al. 2018
        # compute backward scattering taking into account the divergence of the upwelling stream due to refraction
        backward_scattering /= self.effective_permittivity.real
        # ... and subsample to the combined grid
        backward_scattering = _fill_forward(backward_scattering, b_layer)

        # compute the volume backscatter from the backward scattering and extinction
        subgate_dtau = 2 * subgate_layer_extinction * dz  # two-way optical depth of the layer
        # 1st order analytical integration of the backscatter in the subgate grid (where extinction is constant).
        subgate_backscatter_v = (1 - np.exp(-subgate_dtau)) / (2 * subgate_layer_extinction) * backward_scattering

        # now compute the total attenuation, first due to the volume, second due to the interfaces
        # layer 'volume' attenuation: interpolate to z
        subgate_tau_v = np.insert(np.cumsum(subgate_dtau), 0, 0)  # two-way
        # calculate attenuation. Note that np.inf allow to set attenuation to 0 for gates > snowdepth or gates < 0
        subgate_attenuation_v = np.exp(-subgate_tau_v)

        # 'interface' attenuation due to transmission. We neglect incidence angle dependency for the transmission.
        upper_eps = np.insert(self.effective_permittivity[:-1], 0, 1)
        lower_eps = self.effective_permittivity

        # we neglect the diffuse transmission (it must be small compared to the coherent transmission.
        # For GO, use geometric_approximation_backscatter which use a trick, by magically transmitting coherently energy
        # For IEM, no need to use a trick
        transmission = [
            i.coherent_transmission_matrix(self.sensor.frequency, eps_1, eps_2, mu1=1.0, npol=2)[0, 0]
            for i, eps_1, eps_2 in zip(self.snowpack.interfaces, upper_eps, lower_eps)
        ]
        cum_transmission = np.cumprod(np.array(transmission) ** 2, axis=0)  # two-way transmission

        subgate_attenuation_i = np.insert(_fill_forward(cum_transmission, b_layer), 0, 1.0)

        # attenuation at the top of a layer (below the interface) is the product of layer and interface attenuation
        # we now compute the volume backscatter of each subgate
        subgate_backscatter_v *= subgate_attenuation_v[:-1] * subgate_attenuation_i[1:]

        # at last compute the volume backscatter of each gate by computing the primitive of the subgate volume backscatter,
        # select the gate interval and differentitate to get the integrated backscatter over each gate
        subgate_backscatter_v = np.insert(subgate_backscatter_v, 0, 0)
        gate_backscatter_v = np.diff(np.insert(np.cumsum(subgate_backscatter_v)[b_gate], 0, 0))

        ############################################################################################
        # calculate interface echo for each interface (nadir incidence only)
        # this includes a coherent component (Fung and Eom, 1983) and a diffuse component

        def cm(m):  # condition the matrix shape
            m = m.diagonal[0].squeeze()
            if (len(mu_upper_interface[0]) > 1) and np.isscalar(m):
                m = np.full(len(mu_upper_interface[0]), m)
            return m

        flat = Flat()
        # convert to local angle (accounting for small pitch and roll)

        mu_upper_interface = np.sqrt(1 - (1 - mu_i[None, :]) / upper_eps[:, None]).real

        interface_echo = [
            (
                # coherent
                cm(
                    flat.specular_reflection_matrix(self.sensor.frequency, eps_1, eps_2, mu1=mu, npol=2)
                    # use the rms_height of the interface and if not use the "macroscopic" sigma_surface
                    * coherent_reflection_factor(self.sensor, i.roughness_rms, mu)
                ),
                # incoherent
                cm(
                    i.diffuse_reflection_matrix(
                        self.sensor.frequency, eps_1, eps_2, mu_s=mu, mu_i=mu, dphi=np.pi, npol=2
                    )
                )
                / eps_1.real,
            )
            for i, eps_1, eps_2, mu in zip(
                self.snowpack.interfaces,
                upper_eps,
                lower_eps,
                mu_upper_interface,
            )
        ]

        # print([(

        #         # coherent
        #         cm(
        #             flat.specular_reflection_matrix(self.sensor.frequency, eps_1, eps_2, mu1=mu, npol=2)
        #             # use the rms_height of the interface and if not use the "macroscopic" sigma_surface
        #             * coherent_reflection_factor(self.sensor, i.roughness_rms, mu)
        #         ),
        #         # incoherent
        #         cm(
        #             i.diffuse_reflection_matrix(
        #                 self.sensor.frequency, eps_1, eps_2, mu_s=mu, mu_i=mu, dphi=np.pi, npol=2
        #             )/ eps_1.real
        #         )
        #     )
        #     for i, eps_1, eps_2, mu in zip(
        #         self.snowpack.interfaces,
        #         upper_eps,
        #         lower_eps,
        #         mu_upper_interface,
        #     )
        # ])

        # note that the division by eps_1 takes into account the divergence of the upwelling stream due to refraction
        # now add the contribution of the substrate if any
        if self.snowpack.substrate is not None:
            s = self.snowpack.substrate
            mu_last_layer = np.sqrt(1 - (1 - mu_i) / lower_eps[-1].real)
            interface_echo += [
                (
                    cm(
                        flat.specular_reflection_matrix(
                            self.sensor.frequency,
                            lower_eps[-1],
                            s.permittivity(self.sensor.frequency),
                            mu1=mu_last_layer,
                            npol=2,
                        )
                        # use the rms_height of the interface and if not use the "macroscopic" sigma_surface
                        * coherent_reflection_factor(self.sensor, s.roughness_rms, mu_last_layer)
                    ),
                    cm(
                        s.diffuse_reflection_matrix(
                            self.sensor.frequency,
                            lower_eps[-1],
                            mu_s=mu_last_layer,
                            mu_i=mu_last_layer,
                            dphi=np.pi,
                            npol=2,
                        )
                    )
                    / lower_eps[-1].real,
                )
            ]
        else:
            # no echo from the bottom
            zero = np.zeros_like(interface_echo[-1][0])
            interface_echo += [(zero, zero)]

        interface_echo = np.transpose(interface_echo)

        if len(mu_i) > 1:
            # check that the first dimension is that of mu_i
            assert interface_echo.shape[0] == len(mu_i)

        # we don't need subgate_backscatter_i (as opposed to the LRM code) but we need gate_backscatter_i and echo_time
        # subgate_backscatter_i = fill(interface_echo, b_interface) * subgate_attenuation_v * subgate_attenuation_i
        backscatter_i = interface_echo * subgate_attenuation_v[b_interface] * subgate_attenuation_i[b_interface]
        # return the gate volume backscatter and the backscatter of each interface
        return gate_backscatter_v, backscatter_i


def _fill_forward(a, where, axis=-1):
    # Create an array of size where, which is filled with a and fill foward from the beginning of where to the end.
    assert np.array(a).dtype == np.float64
    idx = np.cumsum(where)
    return np.take(np.insert(np.array(a, dtype=np.float64), 0, np.nan, axis=-1), idx, axis=-1)


def delay_doppler_model(model: str | Type, **options) -> Type:
    """Return a delay_doppler_model class based on its name, eventually specializing it if options are provided."""
    return lib.class_specializer("rtsolver.delay_doppler_model", model, **options)


def coherent_reflection_square_decay(sensor):
    # for the sqrt(2),see Claude de Rijke-Thomas thesis, p 106-111.
    beta0 = np.sqrt(C_SPEED / (sensor.pulse_bandwidth * sensor.altitude)) * np.sqrt(2)

    beta12 = 1 / (sensor.wavenumber * sensor.altitude * beta0) ** 2 + beta0**2 / 4
    return beta12


def coherent_reflection_factor(sensor, roughness_rms, mu):
    """Return the factor for the coherent echo due to the spherical wave.

    See Fung and Eom (1983), equation 6. This neglects the macroscopic
    slope of the terrain, which should be included in principle.
    """
    sintheta2 = 1 - mu**2  # we neglect the slope
    theta2 = sintheta2  # approximation

    beta12 = coherent_reflection_square_decay(sensor)

    return np.exp(-4 * (sensor.wavenumber * roughness_rms) ** 2 - theta2 / beta12) / beta12 / (4 * np.pi)


def local_incidence_cosine(sensor, mu):
    """Compute the cosine of the local incidence angle considering small pitch and roll.

    This function assumes pitch and roll are small; otherwise yaw would be involved in the equation.
    """
    return mu * np.cos(sensor.pitch_angle) * np.cos(sensor.roll_angle)
