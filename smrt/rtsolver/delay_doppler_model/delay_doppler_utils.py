from collections.abc import Callable
from typing import Optional

import numba
import numpy as np
import numpy.typing as npt
import scipy.signal

from smrt.core.error import SMRTError, smrt_warn
from smrt.core.globalconstants import C_SPEED
from smrt.core.terrain import TerrainInfo


def delay_sampling_vector(sensor, oversampling_time: int, window_widening: int = 1):
    """Compute the time/delay of each gate, considering oversampling and window_widening

    :param sensor: Sensor instance containing all the required parameters
    :param oversampling_time: factor of oversampling in the time dimension
    :param window_widening: factor of widening the range of times w/r to that recorded by the sensor (multiply ngate in practice)
    """
    # return np.arange(
    #     -sensor.nominal_gate * oversampling_time,
    #     (sensor.ngate * window_widening - sensor.nominal_gate) * oversampling_time,
    # ) / (sensor.pulse_bandwidth * oversampling_time)

    return (
        np.arange(0, sensor.ngate * window_widening * oversampling_time) / oversampling_time - sensor.nominal_gate
    ) / sensor.pulse_bandwidth


def doppler_frequency_vector(sensor, oversampling_doppler: int):
    """Compute the Doppler frequency considering oversampling

    :param sensor: Sensor instance containing all the required parameters
    :param oversampling_time: factor of oversampling in the Doppler frequency dimension
    """
    # return np.linspace(
    #     -sensor.pulse_repetition_frequency / 2,
    #     sensor.pulse_repetition_frequency / 2,
    #     sensor.ndoppler * oversampling_doppler,
    # )

    L_2 = sensor.ndoppler // 2 * oversampling_doppler

    return np.arange(-L_2 + 0.5, +L_2) * (sensor.pulse_repetition_frequency / (oversampling_doppler * sensor.ndoppler))


def check_low_ndoppler(ndoppler, ddm_model):
    if ndoppler < 16:
        smrt_warn(
            f"The DDM model {type(ddm_model).__name__} is not suitable for a low number of doppler frequencies 'ndoppler={ndoppler}'."
            " This is due to approximating the SAR compressed window to be a rectangle, a sinc, or a gaussian."
            " Only models using sin(ndoppler*x)/sin(x) are suitable."
        )


#
# map to convert PTR to gaussian for the models that prefer gaussian PTR.
# Different authors have used different values.
# Otherwise sinc^2 is used by default
#

ptr_gaussian_approximation_map = {
    # J. MacArthur, "Design of the SEASAT-A Radar Altimeter," OCEANS '76, Washington, DC, USA, 1976, pp. 222-229,
    # doi: 10.1109/OCEANS.1976.1154217.
    "gaussian-macarthur76:rect": (0.443, 1.0),
    # J. MacArthur and J. H. U. A. P. Laboratory. SEASAT - a radar altimeter design description. Johns Hopkins
    # University Applied Physics Laboratory, 1978.
    "gaussian-macarthur78:rect": (0.513, 1.0),
    # G. Brown. The average impulse response of a rough surface and its applications. IEEE Trans.
    # Antennas and Propagation, 25(1):67 74, Jan. 197
    "gaussian-brown77:rect": (0.425, 1.0),
    # Parameters for the gaussian equivalent to the sinc + hamming. See Fig 4 in Ray et al. 2015
    "gaussian-ray15:hamming": (0.5408, 1.0055),
    # Parameters from Dinardo et al. 2018, that refer to Fenoglio-Marc et al., 2015a Fenoglio-Marc, L., Dinardo, S.,
    # Scharroo, R., Roland, A., Dutour, Sikiric M., Lucas, B., Becker, M., Benveniste, J., Weiss, R., 2015a. The German
    # bight: a validation of CryoSat-2 altimeter data in SAR mode. Adv. Space Res. 55 (11), 2641–2656.
    # https://doi.org/10.1016/j.asr.2015.02.014.
    "gaussian-dinardo18:hamming": (0.55, 1.0),
    # Value recalculated for SMRT by using least-square
    "gaussian-smrt:hamming": (0.54973121, 1.00758253),
    "gaussian-smrt:rect": (0.36018704, 1.01917462),
    "gaussian-smrt-test:rect": (0.1, 1),
}


def ptr_function(ptr: str) -> Callable[[npt.NDArray, npt.NDArray], npt.NDArray]:
    """Return the function (callable with 2 arguments) to be used for the convolution described by the string ptr

    :param str: name of the function to return.
    """
    if ptr == "sinc^2":
        return sinc2
    else:
        raise NotImplementedError("only sinc2 is currently available but this c/should change")


def ft_ptr_function(ptr: str) -> Callable[[npt.ArrayLike, npt.ArrayLike], np.ndarray]:
    """Return the Fourier transform of the function (callable with 2 arguments) to be used for the convolution described
    by the string ptr. The return value has no unit.

    :param str: name of the function to return.
    """
    if ptr == "sinc^2":
        return triangular_function
    else:
        raise NotImplementedError("only sinc2 is currently available but this c/should change")


def ptr_gaussian_approximation(ptr: str, window: str) -> tuple[float, float]:
    """Return the coefficient for the gaussian approximation of sinc2 or sinc2+hamming

    :param ptr: type of approximation
    :param window: rect or hamming window used in the SAR processing
    """
    if ":" in ptr:
        return ptr_gaussian_approximation_map[ptr]
    else:
        if window is None:
            raise SMRTError("doppler_window must be set to use this DDM model. Can be 'rect', 'hamming', or ...")
        return ptr_gaussian_approximation_map[ptr + ":" + window]


def sinc2(x: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
    return np.sinc(x / b) ** 2


@numba.jit(nopython=True, cache=True)
def triangular_function(x: npt.ArrayLike, a: npt.ArrayLike) -> np.ndarray:
    """The return value as the unit of 1 / a"""
    x = np.asarray(x)
    a = np.asarray(a)
    # see B18 Eq 32 and 33
    f = (1 - np.abs(x / a)) / a
    return np.where(f > 0, f, 0)  # unit of a^-1


def delay_compensation(sensor, delay_doppler_map, oversampling_time, oversampling_doppler, with_earth_curvature=True):
    """Perform the delay compensation on the delay_doppler_map array using sensor characteristics and oversampling in the
    doppler and time dimension.

    """
    idelta_r1 = _compute_idelta_r1(sensor, oversampling_time, oversampling_doppler, with_earth_curvature)
    return _jit_delay_compensation(delay_doppler_map, idelta_r1)


@numba.jit(nopython=True, cache=True)
def _jit_delay_compensation(delay_doppler_map, idelta_r1):
    compensated_delay_doppler_map = np.zeros_like(delay_doppler_map)

    # oversampled ngate and nofdoppler. ngate may also have window widening
    nogate, nofdoppler = delay_doppler_map.shape

    for n in range(nofdoppler):
        idelta_rn = np.floor(idelta_r1 * (n - nofdoppler // 2 + 0.5) ** 2)
        if idelta_rn < nogate:
            compensated_delay_doppler_map[0 : nogate - idelta_rn, n] = delay_doppler_map[idelta_rn:, n]
    return compensated_delay_doppler_map


def _compute_idelta_r1(sensor, oversampling_time, oversampling_doppler, with_earth_curvature):
    f1 = 1 / sensor.burst_duration / oversampling_doppler
    # Eq 19 Halimi et al. 2014
    alpha = sensor.alpha if with_earth_curvature else 1.0

    delta_r1 = f1**2 * (alpha * sensor.altitude * sensor.wavelength**2 / (8 * sensor.velocity**2))  # fmt: skip
    idelta_r1 = delta_r1 * (2 / C_SPEED * sensor.pulse_bandwidth * oversampling_time)
    return idelta_r1


def numerical_convolution(
    FSSR: npt.NDArray,
    ptr_time: Optional[Callable[[npt.ArrayLike, npt.ArrayLike], np.ndarray]],
    ptr_doppler: Optional[Callable[[npt.ArrayLike, npt.ArrayLike], np.ndarray]],
    sensor,
    terrain_info: Optional[TerrainInfo],
    oversampling_time: int,
    oversampling_doppler: int,
):
    """Compute the convolution with the ptr_f, ptr_t, and PDF assuming a gaussian surface. It does not change the unit
    of the ddm if both ptr_time and ptr_doppler are applied.
    TODO: make it generic for any type of surface distribution
    """
    # perform the PTR_f convolution
    if ptr_doppler is not None:
        Nb = sensor.ndoppler * oversampling_doppler
        delta_fn = sensor.pulse_repetition_frequency / Nb  # unit: s^-1
        fn = np.arange(-Nb // 2 + 0.5, +Nb // 2) * delta_fn
        fn = fn[np.newaxis, :]
        ptr_f = ptr_doppler(
            fn, sensor.pulse_repetition_frequency / sensor.ndoppler
        )  # Eq 15  Halimi et al. 2014  # unit: no

        # convolve with the ptr_f
        ddm = scipy.signal.fftconvolve(FSSR, ptr_f, mode="same", axes=1) * delta_fn  # unit: s^-1
    else:
        ddm = FSSR  # unit: no

    if (ptr_time is None) and ((terrain_info is None) or (terrain_info.sigma_surface == 0)):
        return ddm

    # perform the PTR_t and PDF convolution
    nogate_2 = sensor.ngate * oversampling_time // 2

    dtau = 1 / (oversampling_time * sensor.pulse_bandwidth)
    ctau = np.arange(-nogate_2 + 1, nogate_2)[:, np.newaxis] * dtau  # the +1 has been verified

    if ptr_time is not None:
        ptr_t = ptr_time(ctau, 1 / sensor.pulse_bandwidth)  # Eq 6 Halimi et al. 2014  # unit: no

    if (terrain_info is not None) and (terrain_info.sigma_surface > 0):
        # the PDF convolution
        sigma_s = 2 * terrain_info.sigma_surface / C_SPEED  # unit s
        assert terrain_info.distribution == "normal"  # TODO allows more distributions
        pdf = np.exp(-(ctau**2) / (2 * sigma_s**2)) / (
            np.sqrt(2 * np.pi) * sigma_s
        )  # Eq 5 Halimi et al. 2014 # unit: s^-1

        if ptr_time is not None:
            pdf_ptr_t = scipy.signal.fftconvolve(ptr_t, pdf, mode="same") * dtau  # unit: no
        else:
            raise NotImplementedError("not sure this is relevant")
            pdf_ptr_t = pdf * dtau  # unit: no
    else:
        pdf_ptr_t = ptr_t  # unit: no

    # convolve with the ptr_t and PDF.
    ddm = scipy.signal.fftconvolve(ddm, pdf_ptr_t, mode="same", axes=0) * dtau  # Eq 7 Halimi et al. 2014     # unit: no

    return ddm  # unit: no if both ptr_time and ptr_doppler are applied. unit: s if only ptr_time is applied
