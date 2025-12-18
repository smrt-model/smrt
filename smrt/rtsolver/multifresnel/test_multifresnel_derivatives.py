import numpy as np
from smrt.rtsolver.multifresnel.multifresnel_derivatives import complex_polarized_id23


def test_complex_polarized_id23():
    I2 = complex_polarized_id23(2)
    expected_I2 = np.array(
        [[[[1, 1], [1, 1]],
          [[0, 0], [0, 0]],
          [[0, 0], [0, 0]]],
         
         [[[0, 0], [0, 0]],
          [[1, 1], [1, 1]],
          [[0, 0], [0, 0]]]]
    )
    np.testing.assert_allclose(I2, expected_I2)
