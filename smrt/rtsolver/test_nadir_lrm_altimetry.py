import numpy as np
import pytest

from smrt import make_snowpack
from smrt.core.model import Model
from smrt.emmodel.nonscattering import NonScattering
from smrt.emmodel.rayleigh import Rayleigh
from smrt.inputs.altimeter_list import envisat_ra2
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

    assert np.allclose(np.sum(res.waveform()), 4.25624771e-24)
