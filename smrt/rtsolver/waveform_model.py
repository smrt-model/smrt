
import numpy as np
import scipy.special

from smrt.core.globalconstants import C_SPEED, EARTH_RADIUS


class WaveformModel(object):
    pass


class Brown1977(WaveformModel):
    """Antenna Gain formulation used by Brown 1977. The formula is exp(2/gamma * sin(theta)**2) for the perfect nadir case,
but is also available with off-nadir angles.
"""
    __name__ = "brown_1977"

    def __init__(self, sensor, numerical_convolution=False):

        super().__init__()

        self.sensor = sensor

        self.G0 = 1  # antenna_gain0

        log2 = 0.6931471805599453
        self.gamma = 2 / log2 * np.sin(np.deg2rad(self.sensor.beamwidth) / 2)**2

        self.numerical_convolution = numerical_convolution

    def G(self, theta, phi):

        rho_h = np.tan(theta)

        newtheta = np.arccos((np.cos(self.sensor.off_nadir_angle) + rho_h * np.sin(self.sensor.off_nadir_angle)
                              * np.cos(phi)) / np.sqrt(1 + rho_h**2))
        return self.G0 * np.exp(-2 / self.gamma * np.sin(newtheta)**2)

    def PFS(self, tau, surface_slope=0, shift_nominal_gate=True):
        # tau = t - 2*h/c

        # e
        # take into account the nominal_gate shift
        if shift_nominal_gate:
            otau = tau - self.sensor.nominal_gate / self.sensor.pulse_bandwidth
        else:
            otau = tau

        # include Earth curvature as in Newkrik and Brown, 1992
        e = C_SPEED / (self.sensor.altitude * (1 + self.sensor.altitude / EARTH_RADIUS)) * otau  #

        coef = self.G0**2 * self.sensor.wavelength**2 * C_SPEED / (4 * (4 * np.pi)**2 * self.sensor.altitude**3)

        if self.sensor.off_nadir_angle != 0 and surface_slope != 0:
            raise NotImplementedError("It is currently not possible to account for both off_nadir and tilted terrain. It would be necessary"
                                      "to include the azimuths of the slope and the sensor tild and compute the true angle. To avoid this complexity"
                                      "we consider only one can be set.")

        theta = self.sensor.off_nadir_angle + surface_slope  # !! this equation is false if the surface and sensors tilt are not aligned !!

        def negexp(x):
            return np.where(x <= 0, np.exp(x), 0)

        if theta == 0:  # pure nadir
            return coef * negexp(-4 / self.gamma * e)
        else:  # off nadir
            return coef * negexp(-4 / self.gamma * (np.sin(theta)**2 + e * np.cos(2 * theta))) \
                * scipy.special.i0(4 / self.gamma * np.sqrt(e) * np.sin(2 * theta))

    def PFS_PTR_PDF(self, tau, sigma_surface=0, surface_slope=0):
        """compute the convolution of the PFS and PTR

        :param sensor: sensor to apply the antenna gain
        :param tau: time to which to compute the PFSxPTR
        :param sigma_surface: RMS height of the surface topography (meter)
"""

        sqrt2 = 1.4142135623731
        sigma_c = np.sqrt(self.sensor.pulse_sigma**2 + (2 * sigma_surface / C_SPEED)**2)

        pfs = self.PFS(tau, surface_slope=surface_slope, shift_nominal_gate=False)

        otau = (tau - self.sensor.nominal_gate / self.sensor.pulse_bandwidth)

        if sigma_c > 0:

            if self.numerical_convolution:
                sigma_c_sqrt2 = sigma_c * sqrt2
                ptr = np.exp(-(otau / sigma_c_sqrt2)**2)

                return np.convolve(ptr, pfs) / (self.sensor.pulse_bandwidth * np.sum(ptr))
            else:
                i0 = (otau >= 0).argmax()
                pfs[i0:] = pfs[0:-i0]  # shift
                pfs[0:i0] = pfs[i0]

                return pfs * (1 + scipy.special.erf(otau / (sqrt2 * sigma_c))) / 2 / self.sensor.pulse_bandwidth
        else:
            i0 = (otau >= 0).argmax()
            pfs[i0:] = pfs[0:-i0]  # shift
            pfs[0:i0] = 0
            return pfs / self.sensor.pulse_bandwidth


class Newkrik1992(WaveformModel):
    """Antenna Gain formulation proposed by Newkrik and Brown, 1992. Compared to the classical Bronw 1977, it takes into account 
    the asymmetry of the antenna pattern in the co and cross-track direction.

"""

    __name__ = "Newkrik1992"

    def __init__(self, sensor):

        self.sensor = sensor
        self.G0 = 1

        log2 = 0.6931471805599453
        self.gamma = 2 / log2 * np.sin(np.deg2rad(self.sensor.beamwidth) / 2)**2

    def G(self, theta, phi):

        rho_h = np.tan(theta)
        rho0_h = np.tan(self.sensor.off_nadir_angle)

        sin_omega2 = rho_h**2 * np.sin(phi)**2 / (rho_h**2 - 2 * rho_h
            * rho0_h * np.cos(phi) + rho0_h)

        return self.G0 * np.exp(-2 / self.gamma * (1 + self.sensor.beam_asymmetry * sin_omega2**2) * np.sin(theta)**2)

    def PFS(self, sensor, tau):

        # include Earth curvature as in Newkrik and Brown, 1992
        e2 = C_SPEED / (self.sensor.altitude * (1 + self.sensor.altitude / EARTH_RADIUS)) * tau

        return self.G0**2 * self.sensor.wavelength**2 * C_SPEED / (4 * (4 * np.pi)**2 * self.sensor.altitude**3) \
            * np.exp(-4 / self.gamma * e2 * (1 + self.sensor.beam_asymmetry / 2)) \
            * scipy.special.i0(2 * self.sensor.beam_asymmetry / self.gamma * e2)

    #def PFS_PTR_PDF(self, tau, sigma_surface=0, surface_slope=0):
    #    riase NotImplementedError("to be implemented")
