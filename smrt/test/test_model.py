# coding: utf-8


from smrt.rtsolver.dort import DORT
from smrt.inputs.make_medium import make_snowpack
from smrt.core.model import Model

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
