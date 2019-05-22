

import numpy as np
from smrt.substrate.reflector import Reflector


mu = np.cos(np.arange(0, 90))


def test_scalar_specular():

    refl = Reflector(specular_reflection=0.5)

    m = refl.specular_reflection_matrix(None, None, mu, 2)

    assert(np.all(m.diagonal() == 0.5))


def test_dict_specular():

    refl = Reflector(specular_reflection={'H': 0.5, 'V': 0.7})

    m = refl.specular_reflection_matrix(None, None, mu, 2)

    assert(abs(np.mean(m.diagonal()) - 0.6) < 1e-10)
    assert(np.sum(m.diagonal() == 0.5) == len(mu))
    assert(np.sum(m.diagonal() == 0.7) == len(mu))


def test_func_specular():

    def refl(theta):
        return np.full(len(theta), 0.5)

    refl = Reflector(specular_reflection=refl)
    m = refl.specular_reflection_matrix(None, None, mu, 2)

    assert(np.all(m.diagonal() == 0.5))
