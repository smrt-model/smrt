
import numpy as np
from collections.abc import Iterable

from smrt.core.error import SMRTError
from smrt.core.globalconstants import C_SPEED

import matplotlib.pyplot as plt

colors = {'kirchoff': '#87CEEB', 'IEM': '#FF6F61', 'SPM': '#32CD32', 'SSA': '#FFD700', 'GO': '#708090'}

def validity_diagram(sensor=None, snowpack=None, interface=None, rms_height=None, correlation_length=None, frequency=None, ax=None):
    """plot a validity diagram for the rough surface model (assuming a Gaussian surface) with the roughness of the
    snowpack (and/or the interface, or provides RMS and correlation values).

    :param sensor: sensor with a single or multiple frequencies.
    :param snowpack: snowpack from which to take the interfaces.
    :param interface: an interface or a list of interfaces (can be a substrate as well).
    :param rms_height: other rms_height to be plotted.
    :param correlation_length: other correlation_length to be plotted.
    :param frequency: frequency if the sensor is not provided.
    """

    if ax is None:
        plt.figure()
        ax = plt.gca()

    kl = 10**np.linspace(-1, 2, 400)
    ks = 10**np.linspace(-1, 1.5, 400)

    # kirchoff

    # Eq 45: https://www.foi.se/rest-api/report/FOI-R--0988--SE
    # kl > 6
    # Rc > lambda  avec Rc = l**2 sqrt(pi) / (2 * s * sqrt(6))
    # or kRc = kl**2 sqrt(pi) / (2 * * ks * sqrt(6))  and k*lamdba = 2*phi

    coef = np.sqrt(np.pi) / (4 * np.pi * np.sqrt(6))
    kl_kirr = np.sqrt(ks / coef)
    ax.loglog(np.full_like(ks, 6)[kl_kirr <= 6], ks[kl_kirr <= 6], color=colors['kirchoff'])
    ax.loglog(kl_kirr[kl_kirr >= 6], ks[kl_kirr >= 6], color=colors['kirchoff'])

    ax.annotate('Kir.', xy =(20, 1), xycoords='data', color=colors['kirchoff'])

    # IEM

    # Eq 55: https://www.foi.se/rest-api/report/FOI-R--0988--SE
    # ks < 3
    # ks*kl < 1.2 * sqrt(eps)
    eps = 1.3  # approx for snow only !!
    kl_iem = 1.2 * np.sqrt(eps) / ks
    ax.loglog(kl[kl < 1.2 * np.sqrt(eps) / 3], np.full_like(ks, 3)[kl < 1.2 * np.sqrt(eps) / 3], color=colors['IEM'])
    ax.loglog(kl_iem[ks <= 3], ks[ks <= 3], color=colors['IEM'])

    ax.annotate('IEM', xy =(0.8, 0.8), xycoords='data', color=colors['IEM'])

    # SPM

    # ks < 0.3
    # kl > sqrt(2) ks / 0.3

    ks_spm = kl / np.sqrt(2) * 0.3
    ax.loglog(kl[ks_spm < 0.3], ks_spm[ks_spm < 0.3], color=colors['SPM'])
    ax.loglog(kl[kl > np.sqrt(2)], np.full_like(kl, 0.3)[kl > np.sqrt(2)], color=colors['SPM'])

    ax.annotate('SPM', xy =(2.5, 0.15), xycoords='data', color=colors['SPM'])


    # SSA
    # KL >> sqrt(2) * KS / cos(theta)
    ax.loglog(kl, 10 * np.sqrt(2) * ks, color=colors['SSA'])

    ax.annotate('SSA', xy =(2, 3), xycoords='data', color=colors['SSA'])

    # GO
    # kl > 2 * pi  Dierking 1999
    # ks * cos(theta) > pi/2

    ax.loglog(kl[kl > 2 * np.pi], np.full_like(kl, np.pi / 2)[kl > 2 * np.pi], color=colors['GO'])
    ax.loglog(np.full_like(ks, 2 * np.pi)[ks > np.pi / 2], ks[ks > np.pi / 2], color=colors['GO'])

    ax.annotate('GO', xy =(40, 10), xycoords='data', color=colors['GO'])


    #ax.plot()

    ax.set_xlabel('k L')
    ax.set_ylabel('k s')

    # now plot our points

    if interface is None:
        interface = []
    elif not isinstance(interface, Iterable):
        interface = [interface]

    if snowpack is not None:
        interface += snowpack.interfaces + [snowpack.substrate]

    correlation_length = [correlation_length] if isinstance(correlation_length, float) else correlation_length
    correlation_length = list(correlation_length) if correlation_length is not None else []
  
    rms_height = [rms_height] if isinstance(rms_height, float) else rms_height 
    rms_height = list(rms_height) if rms_height is not None else []

    correlation_length += [getattr(i, "corr_length", np.nan) for i in interface]
    rms_height += [getattr(i, "roughness_rms", np.nan) for i in interface]

    [print(f'rougness pair (rms, corr_length) plotted : {rms, lc}') for rms, lc in zip(rms_height, correlation_length)]
    
    if sensor is not None:
        frequency = sensor.frequency
    elif frequency is None:
        raise SMRTError("Either sensor or frequency must be provided")

    if not isinstance(frequency, Iterable):
        frequency = [frequency]

    for freq in frequency:
        k = 2 * np.pi * freq / C_SPEED

        kl_p = k * np.array(correlation_length)
        ks_p = k * np.array(rms_height)
        #print(kl_p, ks_p)

        label = f'{freq * 1e-9} GHz'
        ax.loglog(kl_p, ks_p, 'o', label=label)

    ax.set_xlim((min(np.min(kl), np.min(kl_p)),
                 max(np.max(kl), np.max(kl_p))))
    ax.set_ylim((min(np.min(ks), np.min(ks_p)),
                 max(np.max(ks), np.max(ks_p))))

    ax.legend()
