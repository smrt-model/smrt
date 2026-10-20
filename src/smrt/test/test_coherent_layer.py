import warnings

import pytest

# local import
from smrt import make_model, make_snowpack, sensor_list
from smrt.core.error import SMRTWarning
from smrt.interface.coherent_flat import CoherentFlat, process_coherent_layers


@pytest.fixture
def snowpack_with_small_layer():
    density = [300, 916.7, 400, 500]
    thickness = [0.10, 0.01, 0.20, 1000]
    temperature = 270
    corr_length = [200e-6, 0, 200e-6, 200e-6]

    sp = make_snowpack(thickness, "exponential", density=density, temperature=temperature, corr_length=corr_length)
    return sp


def test_process_coherent_layer(snowpack_with_small_layer):
    emmodel_list = [None] * len(snowpack_with_small_layer.layers)

    effective_permittivity = (3.16 + 0.001j) * snowpack_with_small_layer.profile("density") / 917

    sensor = sensor_list.passive(5e9, 0)
    sp, emmodel_list, effective_permittivity = process_coherent_layers(
        snowpack_with_small_layer, emmodel_list, effective_permittivity, sensor
    )

    nlay = len(snowpack_with_small_layer.layers)
    assert len(sp.layers) == nlay - 1
    assert len(emmodel_list) == nlay - 1
    assert len(effective_permittivity) == nlay - 1
    assert len(sp.interfaces) == nlay - 1
    assert sp.layers[1].thickness == 0.20
    assert isinstance(sp.interfaces[1], CoherentFlat)


def test_snowpack_with_coherent_layer(snowpack_with_small_layer):
    # this test is only here to avoid regression, it is not scientifically validated

    theta = 60
    radiometer = sensor_list.passive(5e9, theta)

    # create the EM Model - Equivalent DMRTML
    m = make_model("iba", "dort", rtsolver_options={"n_max_stream": 64, "process_coherent_layers": True})
    warnings.simplefilter("ignore", category=SMRTWarning)
    res = m.run(radiometer, snowpack_with_small_layer)

    print(res.TbV(), res.TbH())

    # assert abs(res.TbV() - 261.1994214138529) < 1e-4
    # assert abs(res.TbH() - 201.1848483718344) < 1e-4

    # the new values may come from the correction of the bug in dort which limited
    # the streams to the non-total reflection ones. This is not all clear yet...
    # assert abs(res.TbV() - 261.05630770071855) < 1e-4
    # assert abs(res.TbH() - 196.83495992559307) < 1e-4

    # the new values come form the correction of 917->916.7
    assert abs(res.TbV() - 261.06731839819327) < 1e-4
    assert abs(res.TbH() - 196.8984511676495) < 1e-4
