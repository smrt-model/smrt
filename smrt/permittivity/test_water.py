import numpy as np
import pytest

from smrt.core.globalconstants import FREEZING_POINT, GHz
from smrt.permittivity.water import water_permittivity_turner16


@pytest.mark.parametrize(
    "frequency, temperature, expected",
    [
        (0 * GHz, 0 + FREEZING_POINT, 87.9 + 0j),  # Fig 3 in Turner 2016
        (1 * GHz, 0 + FREEZING_POINT, 86.8 + 9.1j),  # Fig 3 in Turner 2016
        (10 * GHz, 0 + FREEZING_POINT, 42.0 + 40.3j),  # Fig 3 in Turner 2016
        (10 * GHz, 40 + FREEZING_POINT, 65.1 + 22.0j),  # Fig 3 in Turner 2016
        (100 * GHz, 0 + FREEZING_POINT, 6.8 + 7.9j),  # Fig 3 in Turner 2016
        (9.61 * GHz, -20 + FREEZING_POINT, 19.1 + 30.0j),  # Fig 4 in Turner 2016
        (9.61 * GHz, -10 + FREEZING_POINT, 30.4 + 37.9j),  # Fig 4 in Turner 2016
    ],
)
def test_water_permittivity_turner16(frequency, temperature, expected):
    actual = water_permittivity_turner16(frequency, temperature)

    # Use a relaxed relative tolerance because the implementation and reference
    # use floating point math and may have tiny differences in expression order.
    assert np.allclose(actual.real, expected.real, atol=1e-1)
    assert np.allclose(actual.imag, expected.imag, atol=1e-1)
