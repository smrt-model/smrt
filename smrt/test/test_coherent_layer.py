
import numpy as np
from nose.tools import ok_

# local import
from smrt import make_snowpack, make_model, sensor_list


def test_snowpack_with_coherent_layer():
    # this test is only here to avoid regression, it is not scientifically validated

    density = [300, 917, 400, 500]
    thickness = [0.10, 0.01, 0.20, 1000]
    temperature = 270
    corr_length = [200e-6, 0, 200e-6, 200e-6]


    theta = 60
    radiometer = sensor_list.passive(5e9, theta)

    sp = make_snowpack(thickness, "exponential",
                       density=density, temperature=temperature,
                       corr_length=corr_length)

    # create the EM Model - Equivalent DMRTML
    m = make_model("iba", "dort", rtsolver_options={'n_max_stream': 64, 'process_coherent_layers': True})

    res = m.run(radiometer, sp)

    print(res.TbV(), res.TbH())

    assert abs(res.TbV() - 261.1994214138529) < 1e-4
    assert abs(res.TbH() - 201.1848483718344) < 1e-4