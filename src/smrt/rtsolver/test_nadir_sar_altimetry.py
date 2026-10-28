import numpy as np
import pytest

import smrt
from smrt import make_interface, make_model, make_snowpack
from smrt.core.terrain import TerrainInfo
from smrt.inputs import altimeter_list
from smrt.rtsolver.delay_doppler_model.landy19 import Landy19
from smrt.rtsolver.nadir_sar_altimetry import coherent_reflection_square_decay, compute_incidence_sampling


@pytest.fixture
def reduced_cryosat2():  # reduce sensor to accelerate the computation.
    sensor = altimeter_list.cryosat2(force_circular_antenna=True)
    sensor.ngate = 8
    sensor.nominal_gate = 3
    sensor.ndoppler = 8
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

    # fixed seed for reproducibility; the run must be sequential, otherwise the DEM is drawn in a worker process
    smrt.core.terrain.random_generator = np.random.default_rng(0)

    m = make_altim_model(
        "landy19", oversampling_time=1, oversampling_doppler=1, delay_doppler_model_options=dict(grid_space=500)
    )
    res = m.run(reduced_cryosat2, onelayer_snowpack, parallel_computation=False)
    print(res.waveform())
    np.testing.assert_allclose(
        res.waveform(),
        np.array(
            [
                3.609642e-21,
                2.329584e-20,
                3.517186e-20,
                2.496338e-20,
                2.326063e-20,
                1.639728e-20,
                4.500541e-22,
                2.551174e-22,
            ]
        ),
        rtol=1e-6,
    )


def test_boy17(reduced_cryosat2, onelayer_snowpack):
    m = make_altim_model(
        "boy17", oversampling_time=1, oversampling_doppler=1, delay_doppler_model_options=dict(grid_space=300)
    )

    m.run(reduced_cryosat2, onelayer_snowpack)


def test_landy19_local_slope_flat_dem(reduced_cryosat2, onelayer_snowpack):
    pytest.importorskip("finufft")
    onelayer_snowpack.terrain_info = TerrainInfo(sigma_surface=0)  # flat DEM

    def run(use_local_slope):
        m = make_altim_model(
            "landy19",
            oversampling_time=1,
            oversampling_doppler=1,
            delay_doppler_model_options=dict(grid_space=500, use_local_slope=use_local_slope),
        )
        return m.run(reduced_cryosat2, onelayer_snowpack, parallel_computation=False).delay_doppler_map()

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
        smrt.core.terrain.random_generator = np.random.default_rng(0)  # same DEM for both runs
        return m.run(reduced_cryosat2, onelayer_snowpack, parallel_computation=False).delay_doppler_map()

    res_cst = run(Landy19ConstantSigma0Func).values
    res_nosigma0 = run(Landy19NoSigma0Func).values

    assert np.any(res_cst > 0)
    np.testing.assert_allclose(res_cst, res_nosigma0, rtol=1e-6)


@pytest.mark.parametrize("mode", ["auto", "uniform", "shared", "coherent"])
def test_incidence_sampling_nodes(mode):
    # the sigma0 functions interpolate with np.interp(-mu, -mu_i, bs): mu_i must be strictly decreasing and start at
    # nadir
    sensor = altimeter_list.cryosat2()
    mu_i = compute_incidence_sampling(mode, sensor, 8, 0.002)
    assert mu_i[0] == 1
    assert np.all(np.diff(mu_i) < 0)

    if mode != "uniform":
        # the coherent lobe exp(-theta**2 / beta**2) is resolved from nadir
        beta = np.sqrt(coherent_reflection_square_decay(sensor))
        theta = np.arccos(np.clip(mu_i, -1, 1))
        assert np.max(np.diff(theta[theta <= 6 * beta])) <= beta / 4


def iem_snowpack(thickness):
    iem = make_interface("iem_fung92", roughness_rms=0.002, corr_length=0.02)
    sp = make_snowpack(
        thickness=thickness,
        microstructure_model="exponential",
        density=[300, 350][: len(thickness)],
        corr_length=200e-6,
        temperature=260,
        interface=[iem] * len(thickness),
    )
    sp.terrain_info = TerrainInfo(sigma_surface=0)
    return sp


def test_landy19_coherent_interfaces_delay(reduced_cryosat2):
    # the coherent DDM of each interface must be delayed as the incoherent DDM of the same interface (#99)
    pytest.importorskip("finufft")
    m = make_altim_model(
        "landy19",
        oversampling_time=4,
        oversampling_doppler=1,
        return_oversampled=True,
        return_contributions="full coherent",
        delay_doppler_model_options=dict(grid_space=500),
    )
    waveform = m.run(reduced_cryosat2, iem_snowpack([0.5, 1]), parallel_computation=False).delay_doppler_map()
    waveform = waveform.sum("doppler_frequency").squeeze()

    def half_power_index(contribution):
        w = waveform.sel(contribution=contribution).values
        assert w.max() > 0
        return int(np.argmax(w >= 0.5 * w.max()))

    delay = {
        component: half_power_index(f"{component} interface 1") - half_power_index(f"{component} interface 0")
        for component in ("incoherent", "coherent")
    }
    # 0.5 m of snow (eps ~ 1.52) delays the second interface by about 4.1 ns, i.e. 5 samples at 4 x 320 MHz
    assert delay["incoherent"] >= 4
    assert delay["coherent"] == delay["incoherent"]


def test_landy19_coherent_lobe_resolved(reduced_cryosat2):
    # once the coherent lobe is resolved, the coherent echo must not depend on theta_inc_sampling (#100)
    pytest.importorskip("finufft")

    def coherent_energy(theta_inc_sampling):
        m = make_altim_model(
            "landy19",
            oversampling_time=1,
            oversampling_doppler=1,
            theta_inc_sampling=theta_inc_sampling,
            return_contributions="full coherent",
            delay_doppler_model_options=dict(grid_space=500),
        )
        ddm = m.run(reduced_cryosat2, iem_snowpack([1]), parallel_computation=False).delay_doppler_map()
        return float(ddm.sel(contribution="coherent interface 0").sum())

    np.testing.assert_allclose(coherent_energy(2), coherent_energy(8), rtol=0.01)
