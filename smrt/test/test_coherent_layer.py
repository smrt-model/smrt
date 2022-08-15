
import numpy as np

# local import
from smrt import make_snowpack, make_model, sensor_list


def test_snowpack_with_coherent_layer():
    # this test is only here to avoid regression, it is not scientifically validated

    density = [300, 916.7, 400, 500]
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

    #assert abs(res.TbV() - 261.1994214138529) < 1e-4
    #assert abs(res.TbH() - 201.1848483718344) < 1e-4

    # the new values may come from the correction of the bug in dort which limited
    # the streams to the non-total reflection ones. This is not all clear yet...
    #assert abs(res.TbV() - 261.05630770071855) < 1e-4
    #assert abs(res.TbH() - 196.83495992559307) < 1e-4

    # the new values come form the correction of 917->916.7
    assert abs(res.TbV() - 261.0633483757312) < 1e-4
    assert abs(res.TbH() - 196.8659636937278) < 1e-4

 