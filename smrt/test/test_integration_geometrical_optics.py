# coding: utf-8

import numpy as np

# local import
from smrt import make_snowpack, make_model, sensor_list, make_soil
from smrt.permittivity.water import water_permittivity


def setup_snowpack(l):

    nl = l // 2  # // Forces integer division
    thickness = np.array([0.1, 0.1] * nl)
    thickness[-1] = 100  # last one is semi-infinit
    p_ex = np.array([5e-5] * l)
    temperature = np.array([250.0, 250.0] * nl)
    density = [200, 400] * nl

    soil = make_soil("geometrical_optics_backscatter", water_permittivity, temperature=273.15, mean_square_slope=1e-2)

    # create the snowpack
    snowpack = make_snowpack(thickness=thickness,
                             microstructure_model="exponential",
                             density=density,
                             temperature=temperature,
                             corr_length=p_ex,
                             substrate=soil)
    return snowpack


def test_geometrical_optics():

    # prepare inputs
    snowpack = setup_snowpack(l=2)

    # create the snowpack
    m = make_model("iba", "dort")

    # create the sensor
    radar = sensor_list.active(13e9, 55)

    # run the model
    res = m.run(radar, snowpack)

    print(res.sigmaVV_dB(), res.sigmaHH_dB())

    assert abs(res.sigmaVV_dB() - -27.35490756934666) < 1e-4
    assert abs(res.sigmaHH_dB() - -27.727715758558222) < 1e-4
 
