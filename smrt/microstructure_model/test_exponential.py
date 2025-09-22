from .exponential import Exponential


def test_constructor():
    Exponential({"corr_length": 0.01, "frac_volume": 0.3})
