import fsinc

from .fsinc_2d import sinc2d


def sinc2d_interp_u(x, y, s, xB, yB, xp, yp):
    """Interpolate the uniform samples s(x) onto xp (which could also non-uniform). If
    the samples s(x) are uniform and satisfy the Nyquist-criterion the signal at `xp` is
    reconstructed perfectly.

    Args:
      x (array, floats): uniform sample points
      y (array, floats): uniform sample points
      s (array, floats): uniform sample values at (x, y)
      xB (float): bandwidth, or sampling frequency (1/dx)
      yB (float): bandwidth, or sampling frequency (1/dy)
      xp (array, floats): points of interpolated signal
      yp (array, floats): points of interpolated signal

    Returns:
      sp (array, floats): interpolated signal at (xp, yp).

    """
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert len(s.shape) == 1

    x, xp = fsinc.zero_offset(x, xp)
    y, yp = fsinc.zero_offset(y, yp)

    x = x * xB
    y = y * yB
    xp = xp * xB
    yp = yp * yB

    return sinc2d(x, y, s, xp, yp, True)
