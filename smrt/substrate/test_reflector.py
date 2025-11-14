import numpy as np
import pytest
from typing import Callable

from smrt.core.error import SMRTError
from smrt.substrate.reflector import Reflector

mu = np.cos(np.arange(0, 90))

@pytest.fixture
def reflector_dict():
    return Reflector(
        specular_reflection={
            (21e9, "H"): 0.5,
            (21e9, "V"): 0.6,
            (36e9, "H"): 0.7,
            (36e9, "V"): 0.8,
        }
    )

@pytest.mark.parametrize("type,specular_reflection,m0",
                         [(float,0.5, 0.5),
                          (dict,{"H": 0.5, "V": 0.7}, 0.7),
                          (Callable, lambda theta: np.full(len(theta), 0.5), 0.5)])
def test_scalar_specular(type,specular_reflection, m0):
    #type included for self-explanatory test
    refl = Reflector(specular_reflection=specular_reflection)

    m = refl.specular_reflection_matrix(None, None, mu, 2)

    assert np.all(m[0] == m0)
    assert np.all(m[1] == 0.5)

def test_dict_multifrequency():
    refl = Reflector(specular_reflection={21e9: 0.5, 36e9: 0.7})
    m1 = refl.specular_reflection_matrix(21e9, None, mu, 2)
    m2 = refl.specular_reflection_matrix(36e9, None, mu, 2)

    assert np.all(m1[0] == 0.5)
    assert np.all(m2[1] == 0.7)


def test_missing_frequency_warning():
    with pytest.raises(SMRTError):
        refl = Reflector(specular_reflection={21e9: 0.5, 36e9: 0.7})
        refl.specular_reflection_matrix(10e9, None, mu, 2)


def test_emissivity_reflectivity_relation(reflector_dict):
    r = reflector_dict.specular_reflection_matrix(21e9, None, mu, 2)
    e = reflector_dict.emissivity_matrix(21e9, None, mu, 2)
    assert np.all((r + e).values == 1.0)


def test_tuple_dict_multifrequency(reflector_dict):
    m1 = reflector_dict.specular_reflection_matrix(21e9, None, mu, 2)
    m2 = reflector_dict.specular_reflection_matrix(36e9, None, mu, 2)

    assert np.all(m1[1] == 0.5)
    assert np.all(m1[0] == 0.6)
    assert np.all(m2[1] == 0.7)
    assert np.all(m2[0] == 0.8)


def test_inverted_reflector_dictionary():
    refl = Reflector(
        specular_reflection={
            ("H", 21e9): 0.5,
            ("V", 21e9): 0.6,
            (36e9, "H"): 0.7,
            (36e9, "V"): 0.8,
        }
    )
    m1 = refl.specular_reflection_matrix(21e9, None, mu, 2)
    m2 = refl.specular_reflection_matrix(36e9, None, mu, 2)

    assert np.all(m1[1] == 0.5)
    assert np.all(m1[0] == 0.6)
    assert np.all(m2[1] == 0.7)
    assert np.all(m2[0] == 0.8)
