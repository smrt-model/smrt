import pytest

from smrt import make_interface, make_model, make_snowpack
from smrt.core.terrain import TerrainInfo
from smrt.inputs import sar_altimeter_list
from smrt.rtsolver.delay_doppler_model.boy17 import Boy17
from smrt.rtsolver.delay_doppler_model.buchhaupt18 import Buchhaupt18
from smrt.rtsolver.delay_doppler_model.dinardo18 import Dinardo18
from smrt.rtsolver.delay_doppler_model.halimi14 import Halimi14
from smrt.rtsolver.delay_doppler_model.landy19 import Landy19
from smrt.rtsolver.delay_doppler_model.ray15 import Ray15
from smrt.rtsolver.delay_doppler_model.wingham04 import Wingham04
from smrt.rtsolver.delay_doppler_model.wingham18 import Wingham18


@pytest.fixture
def reduced_cryosat2_sarm():
    sensor = sar_altimeter_list.cryosat2_sarm(force_circular_antenna=True)
    sensor.ngate = 8
    sensor.nominal_gate = 3
    sensor.ndoppler = 4
    return sensor


@pytest.fixture
def onelayer_snowpack():
    GO = make_interface("geometrical_optics_backscatter", roughness_rms=0.005, corr_length=0.10)

    sp = make_snowpack(
        thickness=[1], microstructure_model="exponential", density=300, corr_length=200e-6, temperature=273, surface=GO
    )
    sp.terrain_info = TerrainInfo(sigma_surface=0.2)
    return sp


def make_altim_model(delay_doppler_model, **kwargs):
    return make_model(
        "iba", "nadir_sar_altimetry", rtsolver_options=dict(delay_doppler_model=delay_doppler_model, **kwargs)
    )


def test_dinardo18(reduced_cryosat2_sarm, onelayer_snowpack):
    m = make_altim_model(Dinardo18)

    m.run(reduced_cryosat2_sarm, onelayer_snowpack)


def test_ray15(reduced_cryosat2_sarm, onelayer_snowpack):
    m = make_altim_model(Ray15, oversampling_time=1, oversampling_doppler=1)

    m.run(reduced_cryosat2_sarm, onelayer_snowpack)


def test_buchhaupt18(reduced_cryosat2_sarm, onelayer_snowpack):
    m = make_altim_model(Buchhaupt18, oversampling_time=1, oversampling_doppler=4)  # oversampling_doppler must be >=4

    m.run(reduced_cryosat2_sarm, onelayer_snowpack)


def test_halimi14(reduced_cryosat2_sarm, onelayer_snowpack):
    m = make_altim_model(Halimi14, oversampling_time=1, oversampling_doppler=1)

    m.run(reduced_cryosat2_sarm, onelayer_snowpack)


def test_wingham04(reduced_cryosat2_sarm, onelayer_snowpack):
    m = make_altim_model(Wingham04, oversampling_time=1, oversampling_doppler=1)

    m.run(reduced_cryosat2_sarm, onelayer_snowpack)


def test_wingham18(reduced_cryosat2_sarm, onelayer_snowpack):
    m = make_altim_model(Wingham18, oversampling_time=1, oversampling_doppler=1)

    m.run(reduced_cryosat2_sarm, onelayer_snowpack)


def test_landy19(reduced_cryosat2_sarm, onelayer_snowpack):
    pytest.importorskip("finufft")
    m = make_altim_model(
        Landy19, oversampling_time=1, oversampling_doppler=1, delay_doppler_model_options=dict(grid_space=500)
    )

    m.run(reduced_cryosat2_sarm, onelayer_snowpack)


def test_boy17(reduced_cryosat2_sarm, onelayer_snowpack):
    m = make_altim_model(
        Boy17, oversampling_time=1, oversampling_doppler=1, delay_doppler_model_options=dict(grid_space=300)
    )

    m.run(reduced_cryosat2_sarm, onelayer_snowpack)
