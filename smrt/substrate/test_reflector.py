
import pytest
import numpy as np

from smrt.core.error import SMRTError
from smrt.substrate.reflector import Reflector


mu = np.cos(np.arange(0, 90))


def test_scalar_specular():

    refl = Reflector(specular_reflection=0.5)

    m = refl.specular_reflection_matrix(None, None, mu, 2)

    assert np.all(m[0] == 0.5)
    assert np.all(m[1] == 0.5)


def test_dict_specular():

    refl = Reflector(specular_reflection={'H': 0.5, 'V': 0.7})

    m = refl.specular_reflection_matrix(None, None, mu, 2)

    assert np.all(m[0] == 0.7)
    assert np.all(m[1] == 0.5)


def test_func_specular():

    def refl(theta):
        return np.full(len(theta), 0.5)

    refl = Reflector(specular_reflection=refl)
    m = refl.specular_reflection_matrix(None, None, mu, 2)

    assert np.all(m[0] == 0.5)
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
        m1 = refl.specular_reflection_matrix(10e9, None, mu, 2)


def test_emissivity_reflectivity_relation():
    refl = Reflector(specular_reflection={(21e9, 'H'): 0.5, (21e9, 'V'): 0.6, (36e9, 'H'): 0.7, (36e9, 'V'): 0.8})
    r = refl.specular_reflection_matrix(21e9, None, mu, 2)
    e = refl.emissivity_matrix(21e9, None, mu, 2)
    assert np.all((r + e).values == 1.)


def test_tuple_dict_multifrequency():

    refl = Reflector(specular_reflection={(21e9, 'H'): 0.5, (21e9, 'V'): 0.6, (36e9, 'H'): 0.7, (36e9, 'V'): 0.8})
    m1 = refl.specular_reflection_matrix(21e9, None, mu, 2)
    m2 = refl.specular_reflection_matrix(36e9, None, mu, 2)

    assert np.all(m1[1] == 0.5)
    assert np.all(m1[0] == 0.6)
    assert np.all(m2[1] == 0.7)
    assert np.all(m2[0] == 0.8)


def test_inverted_reflector_dictionary():
    refl = Reflector(specular_reflection={('H', 21e9): 0.5, ('V', 21e9): 0.6, (36e9, 'H'): 0.7, (36e9, 'V'): 0.8})
    m1 = refl.specular_reflection_matrix(21e9, None, mu, 2)
    m2 = refl.specular_reflection_matrix(36e9, None, mu, 2)

    assert np.all(m1[1] == 0.5)
    assert np.all(m1[0] == 0.6)
    assert np.all(m2[1] == 0.7)
    assert np.all(m2[0] == 0.8)
