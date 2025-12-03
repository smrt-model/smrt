import fsinc
import numpy as np

from .fsinc_1d import sinc1d


def sinc1d_interp_u(x, s, xp):
    """Interpolate the uniform samples s(x) onto xp (which could also non-uniform). If
    the samples s(x) are uniform and satisfy the Nyquist-criterion the signal at `xp` is
    reconstructed perfectly.

    Args:
      x (array, floats): uniform sample points
      s (array, floats): uniform sample values
      xp (array, floats): points of interpolated signal

    Returns:
      sp (array, floats): interpolated signal at xp.

    """
    assert np.max(np.abs(np.diff(x) - (x[1] - x[0]))) < 1.0e-15
    x, xp = fsinc.zero_offset(x, xp)

    B = 1.0 / np.mean(np.diff(x))
    print("bandwidth:", B)

    # x = np.arange(0, x.size, 1)
    x = x * B
    xp = xp * B

    return sinc1d(x, s, xp, True)
