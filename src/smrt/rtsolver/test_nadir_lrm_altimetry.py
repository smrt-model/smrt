import numpy as np
import pytest

from smrt import make_interface, make_model, make_snowpack, make_soil_substrate
from smrt.core.model import Model
from smrt.emmodel.nonscattering import NonScattering
from smrt.emmodel.rayleigh import Rayleigh
from smrt.inputs.altimeter_list import envisat_ra2, saral_altika
from smrt.interface.transparent import Transparent
from smrt.rtsolver.nadir_lrm_altimetry import NadirLRMAltimetry


@pytest.fixture
def setup_nonscattering_snowpack():
    return make_snowpack([100], "homogeneous", density=[300], interface=[Transparent])


@pytest.fixture
def setup_scattering_snowpack():
    return make_snowpack([100], "independent_sphere", density=[300], radius=1e-3, interface=[Transparent])


def test_nonscattering_noabsorption(setup_nonscattering_snowpack):
    sp = setup_nonscattering_snowpack

    sensor = envisat_ra2("Ku")
    m = Model(NonScattering, NadirLRMAltimetry, rtsolver_options=dict(theta_inc_sampling=1))
    res = m.run(sensor, sp)

    assert np.all(res.waveform() == 0)


def test_scattering_noabsorption(setup_scattering_snowpack):
    sp = setup_scattering_snowpack

    sensor = envisat_ra2("Ku")
    m = Model(Rayleigh, NadirLRMAltimetry, rtsolver_options=dict(theta_inc_sampling=1))
    res = m.run(sensor, sp)
    print(np.sum(res.data))

    np.testing.assert_allclose(np.sum(res.waveform()), 4.25624771e-24, atol=1e-8)  # atol is subjective


def test_altika_iem_substrate():
    substrate = make_soil_substrate(
        "iem_fung92", permittivity_model=complex(5, 0.5), temperature=260, roughness_rms=0.005, corr_length=0.05
    )
    surface = make_interface("iem_fung92", roughness_rms=0.005, corr_length=0.05)  # NaN at nadir if mu_i > 1
    sp = make_snowpack(
        [1],
        "exponential",
        density=[300],
        corr_length=[200e-6],
        temperature=[260],
        surface=surface,
        substrate=substrate,
    )

    m = make_model("iba", "nadir_lrm_altimetry")
    res = m.run(saral_altika(), sp)

    assert not np.any(np.isnan(res.waveform()))
