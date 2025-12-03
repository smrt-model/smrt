import finufft as nufft
import numpy as np

from . import fastgl


def sinc2d(x, y, s, xp, yp, norm=False, eps=1.0e-6):
    """Calculate the fast 2D sinc-transform by ways of the non-uniform fast Fourier transform.

    sp = sum sinc(x - xp) * sinc(y - xp) * s

    Args:
      x (array, floats): sample points, x-coordinate
      y (array, floats): sample points, y-coordinate
      s (array, floats or complex): sample values at (x,y)
      xp (array, floats): target grid
      yp (array, floats): target grid
      norm (bool): use normalized sinc: sinc(pi*x)/pi*x (default: False)

    Returns:
      sp (array, floats): transformed signal to (xp, yp)
    """
    # normalized sinc
    if norm:
        x = x * np.pi
        y = y * np.pi
        xp = xp * np.pi
        yp = yp * np.pi

    xm = np.max([np.max(np.abs(x)), np.max(np.abs(xp))])
    ym = np.max([np.max(np.abs(y)), np.max(np.abs(yp))])

    resample = 2  # resample rate
    nx = np.ceil(resample * np.round(xm + 3)).astype("int")
    ny = np.ceil(resample * np.round(ym + 3)).astype("int")

    print("calculate Legendre-Gauss weights (using fastgl), nodes:", nx, "x", ny)
    xx, yy, ww = fastgl.lgwt2d(nx, ny)

    # Fwd FT for signal at xx (G-L nodes)
    h = nufft.nufft2d3(
        x, y, s.astype("complex128"), xx.astype("float64"), yy.astype("float64"), isign=-1, eps=eps, upsampfac=1.25
    )

    # integrate signal using G-L quadrature
    ws = (1.0 / 4.0) * h * ww

    # Inv FT for signal at xx
    sp = nufft.nufft2d3(xx.astype("float64"), yy.astype("float64"), ws, xp, yp, isign=1, eps=eps, upsampfac=1.25)

    if np.all(np.isreal(s)):
        return sp.real
    else:
        return sp


def sincsq2d(x, y, s, xp, yp, norm=False, eps=1.0e-6):
    """Calculate the fast 2D sinc-transform squared by ways of the non-uniform fast Fourier transform.

    sp = sum sinc^2(x - xp) * sinc^2(y - xp) * s

    Args:
      x (array, floats): sample points, x-coordinate
      y (array, floats): sample points, y-coordinate
      s (array, floats or complex): sample values at (x,y)
      xp (array, floats): target grid
      yp (array, floats): target grid
      norm (bool): use normalized sinc: sinc(pi*x)/pi*x (default: False)

    Returns:
      sp (array, floats): transformed signal to (xp, yp)
    """
    # normalized sinc
    if norm:
        x = x * np.pi
        y = y * np.pi
        xp = xp * np.pi
        yp = yp * np.pi

    xm = np.max([np.max(np.abs(x)), np.max(np.abs(xp))])
    ym = np.max([np.max(np.abs(y)), np.max(np.abs(yp))])

    resample = 2  # resample rate
    nx = np.ceil(resample * np.round(xm + 3)).astype("int")
    ny = np.ceil(resample * np.round(ym + 3)).astype("int")

    print("calculate Legendre-Gauss weights (using fastgl), nodes:", 2 * nx, "x", 2 * ny)
    xx, yy, ww = fastgl.lgwt_tri_2d(nx, ny)

    # Fwd FT for signal at xx (G-L nodes)
    h = nufft.nufft2d3(
        x, y, s.astype("complex128"), xx.astype("float64"), yy.astype("float64"), isign=-1, eps=eps, upsampfac=1.25
    )

    # integrate signal using G-L quadrature
    ws = (1.0 / 16.0) * h * ww

    # Inv FT for signal at xx
    sp = nufft.nufft2d3(xx.astype("float64"), yy.astype("float64"), ws, xp, yp, isign=1, eps=eps, upsampfac=1.25)

    if np.all(np.isreal(s)):
        return sp.real
    else:
        return sp
