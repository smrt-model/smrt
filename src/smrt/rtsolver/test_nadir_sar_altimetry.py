import numpy as np
import pytest

from smrt import make_interface, make_model, make_snowpack
from smrt.core.terrain import TerrainInfo
from smrt.inputs import altimeter_list
from smrt.rtsolver.delay_doppler_model.landy19 import Landy19


@pytest.fixture
def reduced_cryosat2():  # reduce sensor to accelerate the computation.
    sensor = altimeter_list.cryosat2(force_circular_antenna=True)
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


def test_dinardo18(reduced_cryosat2, onelayer_snowpack):
    m = make_altim_model("dinardo18")

    m.run(reduced_cryosat2, onelayer_snowpack)


def test_ray15(reduced_cryosat2, onelayer_snowpack):
    m = make_altim_model("ray15", oversampling_time=1, oversampling_doppler=1)

    m.run(reduced_cryosat2, onelayer_snowpack)


def test_buchhaupt18(reduced_cryosat2, onelayer_snowpack):
    m = make_altim_model("buchhaupt18", oversampling_time=1, oversampling_doppler=4)  # oversampling_doppler must be >=4

    m.run(reduced_cryosat2, onelayer_snowpack)


def test_halimi14(reduced_cryosat2, onelayer_snowpack):
    m = make_altim_model("halimi14", oversampling_time=1, oversampling_doppler=1)

    m.run(reduced_cryosat2, onelayer_snowpack)


def test_wingham04(reduced_cryosat2, onelayer_snowpack):
    m = make_altim_model("wingham04", oversampling_time=1, oversampling_doppler=1)

    m.run(reduced_cryosat2, onelayer_snowpack)


def test_wingham18(reduced_cryosat2, onelayer_snowpack):
    m = make_altim_model("wingham18", oversampling_time=1, oversampling_doppler=1)

    m.run(reduced_cryosat2, onelayer_snowpack)


def test_landy19(reduced_cryosat2, onelayer_snowpack):
    pytest.importorskip("finufft")
    m = make_altim_model(
        "landy19", oversampling_time=1, oversampling_doppler=1, delay_doppler_model_options=dict(grid_space=500)
    )

    m.run(reduced_cryosat2, onelayer_snowpack)


def test_boy17(reduced_cryosat2, onelayer_snowpack):
    m = make_altim_model(
        "boy17", oversampling_time=1, oversampling_doppler=1, delay_doppler_model_options=dict(grid_space=300)
    )

    m.run(reduced_cryosat2, onelayer_snowpack)


def test_landy19_local_slope_flat_dem(reduced_cryosat2, onelayer_snowpack):
    pytest.importorskip("finufft")

    def run(use_local_slope):
        m = make_altim_model(
            "landy19",
            oversampling_time=1,
            oversampling_doppler=1,
            delay_doppler_model_options=dict(grid_space=500, use_local_slope=use_local_slope),
        )
        return m.run(reduced_cryosat2, onelayer_snowpack).delay_doppler_map()

    ddm_slope = run(True)
    ddm_noslope = run(False)

    np.testing.assert_allclose(ddm_slope.values, ddm_noslope.values, rtol=1e-6)


def test_landy19_sigma0func_vs_constant_sigma0(reduced_cryosat2, onelayer_snowpack):
    pytest.importorskip("finufft")

    class Landy19ConstantSigma0Func(Landy19):
        # sigma0func branch, with sigma0 functions frozen to their nadir value
        def delay_doppler_map_with_sigma0func(self, terrain_info, sigma0=1):
            sigma0 = [float(s(np.array([1.0]))[0]) if callable(s) else s for s in sigma0]
            return super().delay_doppler_map_with_sigma0func(terrain_info=terrain_info, sigma0=sigma0)

    class Landy19NoSigma0Func(Landy19):
        # hide the sigma0func method to force the constant-sigma0 branch of the solver
        def __getattribute__(self, name):
            if name == "delay_doppler_map_with_sigma0func":
                raise AttributeError(name)
            return super().__getattribute__(name)

        def delay_doppler_map(self, terrain_info):
            return Landy19.delay_doppler_map_with_sigma0func(self, terrain_info=terrain_info, sigma0=1)

    def run(ddm_class):
        m = make_altim_model(
            ddm_class,
            oversampling_time=1,
            oversampling_doppler=1,
            delay_doppler_model_options=dict(grid_space=500),
        )
        return m.run(reduced_cryosat2, onelayer_snowpack).delay_doppler_map()

    np.testing.assert_allclose(run(Landy19ConstantSigma0Func).values, run(Landy19NoSigma0Func).values, rtol=1e-6)


def test_landy19_coherent_interfaces_delay():
    # With IEM interfaces the coherent components are computed with sigma0 functions (sigma0func branch). Their DDMs
    # must be delayed by the propagation time to their own interface, as the incoherent DDMs. The full-size sensor is
    # used because the reduced sensor gives a zero DDM for the interfaces.
    pytest.importorskip("finufft")
    iem = make_interface("iem_fung92", roughness_rms=0.002, corr_length=0.02)
    sp = make_snowpack(
        thickness=[0.5, 1],
        microstructure_model="exponential",
        density=[300, 350],
        corr_length=200e-6,
        temperature=260,
        interface=[iem, iem],
    )
    sp.terrain_info = TerrainInfo(sigma_surface=0)

    m = make_altim_model(
        "landy19",
        oversampling_time=4,
        oversampling_doppler=1,
        return_oversampled=True,
        return_contributions="full coherent",
        delay_doppler_model_options=dict(grid_space=200),
    )
    ddm = m.run(altimeter_list.cryosat2(), sp).delay_doppler_map().sum("doppler_frequency").squeeze()

    def half_power_index(contribution):
        waveform = ddm.sel(contribution=contribution).values
        assert waveform.max() > 0
        return int(np.argmax(waveform >= 0.5 * waveform.max()))

    for i in range(2):
        assert abs(half_power_index(f"coherent interface {i}") - half_power_index(f"incoherent interface {i}")) <= 1
    # 0.5 m of snow (eps ~ 1.52) delays the second interface by about 4.1 ns, i.e. 5 samples at 4 x 320 MHz
    assert half_power_index("incoherent interface 1") - half_power_index("incoherent interface 0") >= 4
