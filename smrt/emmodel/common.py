
import numpy as np


def rayleigh_scattering_matrix_and_angle_tsang00(mu_s, mu_i, dphi, npol=2):
    """compute the Rayleigh matrix and half scattering angle. Based on Tsang theory and application p271 Eq 7.2.16

"""

    mu_s = np.atleast_1d(mu_s)[np.newaxis, :, np.newaxis]
    mu_i = np.atleast_1d(mu_i)[np.newaxis, np.newaxis, :]

    sin_i = np.sqrt(1. - mu_i**2)
    sin_s = np.sqrt(1. - mu_s**2)

    dphi = np.atleast_1d(dphi)
    sinphi = np.sin(dphi)[:, np.newaxis, np.newaxis]
    cosphi = np.cos(dphi)[:, np.newaxis, np.newaxis]

    # Tsang theory and application p127 Eq 3.2.47
    fvv = cosphi * mu_s * mu_i + sin_s * sin_i
    fhv = - sinphi * mu_i   # there is an error in Tsang book. By symmetry of the equation fvh has sin(phi_s-phi_i)
    # and fhv has sin(phi_i-phi_s)
    fhh = cosphi
    fvh = sinphi * mu_s

    if npol == 2:
        p = [[fvv * fvv, fvh * fvh],
             [fhv * fhv, fhh * fhh]]

    elif npol == 3:
        p = [[fvv * fvv, fvh * fvh, fvh * fvv],
             [fhv * fhv, fhh * fhh, fhv * fhh],
             [2 * fvv * fhv, 2 * fvh * fhh, fvv * fhh + fvh * fhv]]
    else:
        raise RuntimeError("invalid number of polarisation")

    # broadcast, it is not automatic (anymore?)
    shape = dphi.size, mu_s.size, mu_i.size
    p = np.array([[np.broadcast_to(ppp, shape) for ppp in pp] for pp in p])

    # compute scattering angle

    cosT = np.clip(mu_s * mu_i + sin_s * sin_i * cosphi, -1.0, 1.0)  # Prevents occasional numerical error

    sin_half_scatt = np.sqrt(0.5 * (1 - cosT))   # compute half the scattering angle

    return p, sin_half_scatt


def rayleigh_scattering_matrix_and_angle_maetzler06(mu_s, mu_i, dphi, npol=2):
    """compute the Rayleigh matrix and half scattering angle. Based on Mätzler 2006 book p111.
This version is relatively slow because it uses phase matrix rotations which is unnecessarily complex for the Rayleigh phase matrix
but would be of interest for other phase matrices.

"""

    # cos and sin of scattering and incident angles in the main frame
    cos_ti = np.atleast_1d(mu_i)[np.newaxis, np.newaxis, :]
    sin_ti = np.sqrt(1. - cos_ti**2)

    cos_t = np.atleast_1d(mu_s)[np.newaxis, :, np.newaxis]
    sin_t = np.sqrt(1. - cos_t**2)

    dphi = np.atleast_1d(dphi)
    cos_pd = np.cos(dphi)[:, np.newaxis, np.newaxis]
    sin_pd_sign = np.where(dphi >= np.pi, -1, 1)[:, np.newaxis, np.newaxis]

    # Scattering angle in the 1-2 frame
    cosT = np.clip(cos_t * cos_ti + sin_t * sin_ti * cos_pd, -1.0, 1.0)  # Prevents occasional numerical error
    cosT2 = cosT**2  # cos^2 (Theta)
    sinT = np.sqrt(1. - cosT2)

    # Apply non-zero scattering denominator
    nonnullsinT = sinT >= 1e-6

    # Create arrays of rotation angles
    cost_sinti = cos_t * sin_ti
    costi_sint = cos_ti * sin_t

    cos_i1 = cost_sinti - costi_sint * cos_pd
    np.divide(cos_i1, sinT, where=nonnullsinT, out=cos_i1)
    np.clip(cos_i1, -1.0, 1.0, out=cos_i1)

    cos_i2 = costi_sint - cost_sinti * cos_pd
    np.divide(cos_i2, sinT, where=nonnullsinT, out=cos_i2)
    np.clip(cos_i2, -1.0, 1.0, out=cos_i2)

    # Special condition if theta and theta_i = 0 to preserve azimuth dependency
    dege_dphi = np.broadcast_to((sin_t < 1e-6) & (sin_ti < 1e-6), cos_i1.shape)
    cos_i1[dege_dphi] = 1.
    cos_i2[dege_dphi] = np.broadcast_to(cos_pd, cos_i2.shape)[dege_dphi]

    # # See Matzler 2006 pg 111 Eq. 3.20
    # # Calculate rotation angles alpha, alpha_i
    # # Convention follows Matzler 2006, Thermal Microwave Radiation, p111, eqn 3.20

    Li = Lmatrix(cos_i1, -sin_pd_sign, (3, npol))    # L (-i1)

    if npol == 2:
        RLi = np.array([[cosT2 * Li[0][0], cosT2 * Li[0][1]],
                        Li[1], [cosT * Li[2][0], cosT * Li[2][1]]])

    elif npol == 3:
        RLi = np.array([[cosT2 * Li[0][0], cosT2 * Li[0][1], cosT2 * Li[0][2]],
                        Li[1], [cosT * Li[2][0], cosT * Li[2][1], cosT * Li[2][2]]])
    else:
        raise RuntimeError("invalid value of npol")

    Ls = Lmatrix(-cos_i2, sin_pd_sign, (npol, 3))    # L (pi - i2)
    p = np.einsum('ij...,jk...->ik...', Ls, RLi)   # multiply the outer dimension (=polarization)

    sin_half_scatt = np.sqrt(0.5 * (1 - cosT))   # compute half the scattering angle

    return p, sin_half_scatt


def Lmatrix(cos_phi, sin_phi_sign, npol):

    # Calculate arrays of rotated phase matrix elements
    # Shorthand to make equations shorter & marginally faster to compute
    cos2_phi = cos_phi**2  # cos^2 (phi)
    sin2_phi = 1 - cos2_phi  # sin^2 (phi)

    sin_2phi = 2 * cos_phi * np.sqrt(sin2_phi)  # sin(2 phi_i)
    sin_2phi *= sin_phi_sign

    if npol == (2, 3):
        s05 = 0.5 * sin_2phi
        L = [[cos2_phi, sin2_phi, s05],
             [sin2_phi, cos2_phi, -s05]]
    elif npol == (3, 2):
        L = [[cos2_phi, sin2_phi],
             [sin2_phi, cos2_phi],
             [-sin_2phi, sin_2phi]]
    else:  # 3 pol
        s05 = 0.5 * sin_2phi
        cos_2phi = 2 * cos2_phi - 1  # cos(2 alpha)
        L = [[cos2_phi, sin2_phi, s05],
             [sin2_phi, cos2_phi, -s05],
             [-sin_2phi, sin_2phi, cos_2phi]]
    return L


# select the faster version
# the equality is not perfect when theta=0 and theta_s=0

rayleigh_scattering_matrix_and_angle = rayleigh_scattering_matrix_and_angle_tsang00
