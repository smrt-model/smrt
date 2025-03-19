# coding: utf-8

import numpy as np
import xarray as xr

import pytest

from smrt.inputs.make_medium import make_snowpack
from smrt.core.model import Model, make_model, make_rtsolver
from smrt.core.result import PassiveResult
from smrt.rtsolver.dort import DORT
from smrt.emmodel.dmrt_qca_shortrange import DMRT_QCA_ShortRange
from smrt.emmodel.dmrt_qcacp_shortrange import DMRT_QCACP_ShortRange
from smrt.core.error import SMRTError

from smrt.inputs.sensor_list import amsre
from smrt.microstructure_model.sticky_hard_spheres import StickyHardSpheres


@pytest.fixture
def onelayer_snowpack():
    # ### Make a snow layer
    sp = make_snowpack([2000], StickyHardSpheres, density=[250], temperature=265, radius=0.3e-3, stickiness=0.2)
    return sp

temperatures = [200, 250, 270]

@pytest.fixture
def onelayer_snowpack_sequence():
    return [
        make_snowpack([2000], StickyHardSpheres, density=[250], temperature=t, radius=0.3e-3, stickiness=0.2)
        for t in temperatures
    ]


def test_multifrequency(onelayer_snowpack):
    m = Model('dmrt_qcacp_shortrange', DORT)

    sensor = amsre()

    m.run(sensor, onelayer_snowpack)


def test_emmodel_dictionary():
    m = Model({'medium1': 'dmrt_qcacp_shortrange', 'medium2': 'dmrt_qca_shortrange'}, DORT)

    sensor = amsre('19')
    snowpacks = make_snowpack(
        [1, 1],
        medium=['medium1', 'medium2'],
        microstructure_model=StickyHardSpheres,
        density=250,
        radius=0.3e-3,
        stickiness=0.2,
    )

    emmodels = m.prepare_emmodels(sensor, snowpacks)

    assert len(emmodels) == 2
    assert isinstance(emmodels[0], DMRT_QCACP_ShortRange)
    assert isinstance(emmodels[1], DMRT_QCA_ShortRange)


def test_joblib_parallel_run(onelayer_snowpack_sequence):
    m = Model('dmrt_qcacp_shortrange', DORT)

    sensor = amsre()

    m.run(sensor, onelayer_snowpack_sequence, parallel_computation=True)


def test_snowpack_dimension(onelayer_snowpack_sequence):
    m = Model('dmrt_qcacp_shortrange', DORT)

    sensor = amsre()

    res = m.run(sensor, onelayer_snowpack_sequence, snowpack_dimension=('temperature', temperatures))

    assert hasattr(res, "temperature")
    assert np.allclose(res.temperature, temperatures)

    with pytest.raises(SMRTError):
        m.run(sensor, onelayer_snowpack_sequence, snowpack_dimension=(temperatures, 'temperature'))


class FakeRTSolver(object):
    def __init__(self, x=0):
        self.x = x

    def solve(self, *args):
        return PassiveResult(xr.DataArray(self.x))


def test_make_model_options(onelayer_snowpack):
    m = make_model('dmrt_qcacp_shortrange', FakeRTSolver, rtsolver_options=dict(x=1))

    sensor = amsre()
    res = m.run(sensor, onelayer_snowpack)

    assert np.all(res.data == 1)


def test_make_model_options_alternative(onelayer_snowpack):
    m = make_model('dmrt_qcacp_shortrange', make_rtsolver(FakeRTSolver, x=1))

    sensor = amsre()
    res = m.run(sensor, onelayer_snowpack)

    assert np.all(res.data == 1)
