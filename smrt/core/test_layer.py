
from nose.tools import raises
from nose.tools import eq_

from .layer import make_microstructure_model
from .error import SMRTError



def test_microstructure_model():
    shs = make_microstructure_model("sticky_hard_spheres", radius=1.0, stickiness=0.5, frac_volume=0.3)


