import finufft as nufft
import numpy as np

from . import fastgl


def sinc1d(x, s, xp, norm=False, eps=1.0e-6):
    """Calculate the fast sinc-transform by ways of the non-uniform fast Fourier transform.

    sp = sum sinc(x - xp) * s

    Args:
      x (array, floats): sample points
      s (array, floats or complex): sample values at x
      xp (array, floats): target grid
      norm (bool): use normalized sinc: sinc(pi*x)/pi*x (default: False)

    Returns:
      sp (array, floats): transformed signal to xp
    """
    # normalized sinc
    if norm:
        x = x * np.pi
        xp = xp * np.pi

    xm = np.max([np.max(np.abs(x)), np.max(np.abs(xp))])

    resample = 2  # resample rate
    nx = np.ceil(resample * np.round(xm + 3)).astype("int")

    # commented by Ghislain
    # print('calculate Legendre-Gauss weights (using fastgl), nodes:', nx)
    xx, ww = fastgl.lgwt(nx)

    # Fwd FT for h = signal at xx (G-L nodes)
    # astypes needed to stop finufft complaining
    h = nufft.nufft1d3(x, s.astype("complex128"), xx.astype("float64"), isign=-1, eps=eps, upsampfac=1.25)

    # integrate signal using G-L quadrature
    ws = h * ww

    # Inv FT for sp = signal at xx
    sp = nufft.nufft1d3(xx.astype("float64"), ws, xp, isign=1, eps=eps, upsampfac=1.25)
    sp = 0.5 * sp

    if np.all(np.isreal(s)):
        return sp.real
    else:
        return sp


def sincsq1d(x, s, xp, norm=False, eps=1.0e-6):
    """Calculate the fast sinc^2-transform by ways of the non-uniform fast Fourier transform.

    sp = sum sinc^2(x - xp) * s

    > This uses Gauss-Legendre quadrature.

    Args:
      x (array, floats): sample points
      s (array, floats or complex): sample values at x
      xp (array, floats): target grid
      norm (bool): use normalized sinc: sinc(pi*x)/pi*x (default: False)

    Returns:
      sp (array, floats): transformed signal to xp
    """
    assert len(x) == len(s)

    # normalized sinc
    if norm:
        x = x * np.pi
        xp = xp * np.pi

    xm = np.max([np.max(np.abs(x)), np.max(np.abs(xp))])

    resample = 2  # resample rate
    nx = np.ceil(resample * np.round(xm + 3)).astype("int")

    # calculate Legendre-Gauss quadrature weights
    # commented by Ghislain
    # print('calculate Legendre-Gauss weights (using fastgl):', nx)
    xx, ww = fastgl.lgwt_tri(nx)

    # Fwd FT for signal at xx
    h = nufft.nufft1d3(x, s.astype("complex128"), xx.astype("float64"), isign=-1, eps=eps, upsampfac=1.25)

    # integrated signal
    ws = 0.25 * h * ww

    # Inv FT for signal at xp
    sp = nufft.nufft1d3(xx.astype("float64"), ws, xp, isign=1, eps=eps, upsampfac=1.25)

    if np.all(np.isreal(s)):
        return sp.real
    else:
        return sp
