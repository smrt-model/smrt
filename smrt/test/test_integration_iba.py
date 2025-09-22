# coding: utf-8


import warnings

import numpy as np
import numpy.testing as npt
import pytest

# local import
from smrt import make_model, make_snowpack, sensor_list


def setup_snowpack(l):
    nl = l // 2  # // Forces integer division
    thickness = np.array([0.1, 0.1] * nl)
    thickness[-1] = 100  # last one is semi-infinit
    p_ex = np.array([5e-5] * l)
    temperature = np.array([250.0, 250.0] * nl)
    density = [200, 400] * nl

    # create the snowpack
    snowpack = make_snowpack(
        thickness=thickness,
        microstructure_model="exponential",
        density=density,
        temperature=temperature,
        corr_length=p_ex,
    )
    return snowpack


def test_iba_dort_oneconfig_passive():
    # prepare inputs
    snowpack = setup_snowpack(l=2)

    for method, atol in {"eig": 1e-4, "shur": 1e-4, "half_rank_eig": 1e-4, "stamnes88": 0.4}.items():
        print("method: ", method)
        m = make_model("iba", "dort", rtsolver_options=(dict(diagonalization_method=method)))

        # create the sensor
        radiometer = sensor_list.amsre("37V")

        # run the model
        if method == "stamnes88":
            warnings.filterwarnings("ignore", ".*not fully validated.*")

        res = m.run(radiometer, snowpack)

        print(res.TbV(), res.TbH())
        # absorption with effective permittivity
        # abs(res.TbV() - 248.08794944809972) < 1e-4
        # abs(res.TbH() - 237.3056263719142) < 1e-4

        npt.assert_allclose(res.TbV(), 248.08374547409588, atol=atol)
        npt.assert_allclose(res.TbH(), 237.30435496083572, atol=atol)


def test_iba_dort_oneconfig_active():
    # prepare inputs
    # create the snowpack
    snowpack = setup_snowpack(l=2)

    for method, atol in {"eig": 1e-3, "shur": 1e-4, "half_rank_eig": 1e-3, "stamnes88": 1.2}.items():
        print("method: ", method)
        m = make_model("iba", "dort", rtsolver_options=(dict(diagonalization_method=method)))

        # create the sensor
        radar = sensor_list.active(frequency=19e9, theta_inc=55)

        # run the model
        if method == "stamnes88":
            warnings.filterwarnings("ignore", ".*not fully validated.*")

        res = m.run(radar, snowpack)

        print(res.sigmaVV_dB(), res.sigmaHH_dB(), res.sigmaHV_dB())

        npt.assert_allclose(res.sigmaVV_dB(), -24.044882546524693, atol=atol)
        npt.assert_allclose(res.sigmaHH_dB(), -24.416295329469907, atol=atol)
        npt.assert_allclose(res.sigmaHV_dB(), -51.544272924876886, atol=atol)


def test_iba_successive_order_oneconfig_passive():
    # prepare inputs
    snowpack = setup_snowpack(l=2)

    m = make_model("iba", "successive_order")

    # create the sensor
    radiometer = sensor_list.amsre("37V")

    res = m.run(radiometer, snowpack)

    print(res.TbV(order="total"), res.TbH(order="total"))

    # values for DORT:
    npt.assert_allclose(res.TbV(order="total"), 248.08374547409588, atol=2e-2)
    npt.assert_allclose(res.TbH(order="total"), 237.30435496083572, atol=2e-2)
    # values for successive order
    npt.assert_allclose(res.TbV(order="total"), 248.06474685001407, atol=1e-3)
    npt.assert_allclose(res.TbH(order="total"), 237.2922307432923, atol=1e-3)


def test_iba_successive_order_oneconfig_active():
    # prepare inputs
    # create the snowpack
    snowpack = setup_snowpack(l=2)

    m = make_model("iba", "successive_order")

    # create the sensor
    radar = sensor_list.active(frequency=19e9, theta_inc=55)

    res = m.run(radar, snowpack)

    print(res.sigmaVV_dB(order="total"), res.sigmaHH_dB(order="total"), res.sigmaHV_dB(order="total"))

    # values for DORT with large tolerance
    npt.assert_allclose(res.sigmaVV_dB(order="total"), -24.044882546524693, atol=2e-1)
    npt.assert_allclose(res.sigmaHH_dB(order="total"), -24.416295329469907, atol=2e-1)
    npt.assert_allclose(res.sigmaHV_dB(order="total"), -51.544272924876886, atol=2)


@pytest.mark.skip("symmetrization is not ready yet")
def test_iba_oneconfig_active_symmetrization():
    # prepare inputs
    # create the snowpack
    snowpack = setup_snowpack(l=2)

    m0 = make_model("iba", "dort", rtsolver_options=(dict(phase_symmetrization=False)))
    m1 = make_model("iba", "dort", rtsolver_options=(dict(phase_symmetrization=True)))

    # create the sensor
    radar = sensor_list.active(frequency=19e9, theta_inc=55)

    res0 = m0.run(radar, snowpack)
    res1 = m1.run(radar, snowpack)

    print(res0.sigmaVV_dB(), res0.sigmaHH_dB(), res0.sigmaHV_dB())
    print(res1.sigmaVV_dB(), res1.sigmaHH_dB(), res1.sigmaHV_dB())

    npt.assert_allclose(res0.sigmaVV_dB(), res1.sigmaVV_dB())
    npt.assert_allclose(res0.sigmaHH_dB(), res1.sigmaHH_dB())
    npt.assert_allclose(res0.sigmaHV_dB(), res1.sigmaHV_dB())


@pytest.mark.skip("symmetrization is not ready yet")
def test_iba_oneconfig_passive_symmetrization():
    # prepare inputs
    # create the snowpack
    snowpack = setup_snowpack(l=2)

    m0 = make_model("iba", "dort", rtsolver_options=(dict(phase_symmetrization=False)))
    m1 = make_model("iba", "dort", rtsolver_options=(dict(phase_symmetrization=True)))

    # create the sensor
    radiometer = sensor_list.amsre("19V")

    res0 = m0.run(radiometer, snowpack)
    res1 = m1.run(radiometer, snowpack)

    print(res0.TbV(), res0.TbH())
    print(res1.TbV(), res1.TbH())

    npt.assert_allclose(res0.TbV(), res1.TbV())
    npt.assert_allclose(res0.TbH(), res1.TbH())
