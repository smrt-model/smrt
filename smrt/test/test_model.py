# coding: utf-8

import numpy as np

import pytest

from smrt.rtsolver.dort import DORT
from smrt.inputs.make_medium import make_snowpack
from smrt.core.model import Model
from smrt.core.error import SMRTError

from smrt.inputs.sensor_list import amsre
from smrt.microstructure_model.sticky_hard_spheres import StickyHardSpheres


def setup_snowpack():
    # ### Make a snow layer
    sp = make_snowpack([2000], StickyHardSpheres, density=[250], temperature=265, radius=0.3e-3, stickiness=0.2)
    return sp


def test_multifrequency():

    m = Model("dmrt_qcacp_shortrange", DORT)

    sensor = amsre()
    testpack = setup_snowpack()

    m.run(sensor, testpack)


def test_joblib_parallel_run():

    m = Model("dmrt_qcacp_shortrange", DORT)

    sensor = amsre()
    snowpacks = [make_snowpack([2000], StickyHardSpheres, density=[250], temperature=t, radius=0.3e-3, stickiness=0.2)
    	for t in [200, 250, 270]]

    m.run(sensor, snowpacks, parallel_computation=True)


def test_snowpack_dimension():

    m = Model("dmrt_qcacp_shortrange", DORT)

    temperatures = [200, 250, 270]

    sensor = amsre()
    snowpacks = [make_snowpack([2000], StickyHardSpheres, density=[250], temperature=t, radius=0.3e-3, stickiness=0.2)
        for t in temperatures]

    res = m.run(sensor, snowpacks, snowpack_dimension=('temperature', temperatures))

    assert np.allclose(res.temperature, temperatures)

    with pytest.raises(SMRTError):
        m.run(sensor, snowpacks, snowpack_dimension=(temperatures, 'temperature'))
