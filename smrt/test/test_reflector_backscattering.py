

import numpy as np

from smrt import make_snowpack, make_model, sensor_list
from smrt.utils import invdB

from smrt.substrate.reflector_backscatter import make_reflector


def test_reflector_backscattering():

    # prepare inputs
    density = [0.01]
    temperature = [210.0]
    thickness = [0.1]

    theta = np.arange(5, 65, 1)
    radar = sensor_list.active(1e9, theta)

    backscattering_coefficient = {'VV': invdB(-10), 'HH': invdB(-15)}

    # Substrate: same for both sample locations. Permittivity is irrelevant parameter
    backscatter_reflector = make_reflector(specular_reflection=0, 
                                           backscattering_coefficient=backscattering_coefficient)

    snowpack = make_snowpack(thickness, "homogeneous",
                            density=density,
                            temperature=temperature,
                            substrate=backscatter_reflector)

    # create the EM Model
    m = make_model("nonscattering", "dort", rtsolver_options=dict(m_max=10))

    res = m.run(radar, snowpack)
    print(res.sigmaVV_dB())
    print(res.sigmaHH_dB())
    assert np.all(np.abs(res.sigmaVV_dB() - (-10)) < 0.01)
    assert np.all(np.abs(res.sigmaHH_dB() - (-15)) < 0.01)

    # must work for any m_max
    m = make_model("nonscattering", "dort", rtsolver_options=dict(m_max=0))

    res = m.run(radar, snowpack)
    print(res.sigmaVV_dB())
    print(res.sigmaHH_dB())
    assert np.all(np.abs(res.sigmaVV_dB() - (-10)) < 0.01)
    assert np.all(np.abs(res.sigmaHH_dB() - (-15)) < 0.01)
