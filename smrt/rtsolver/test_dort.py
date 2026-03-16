import numpy as np
import numpy.testing as npt
import pytest

from smrt import make_snowpack
from smrt.core.model import Model
from smrt.core.sensor import active
from smrt.emmodel.iba import IBA
from smrt.emmodel.rayleigh import Rayleigh
from smrt.rtsolver.dort import DORT, symmetrize_phase_matrix


@pytest.mark.parametrize(
    "microstructure_model,m_max,emmodel,diagonalization_method",
    [("independent_sphere", 6, Rayleigh, "shur"), ("exponential", 16, IBA, "shur_forcedtriu")],
)
def test_shur_based_diagonalisation(microstructure_model, m_max, emmodel, diagonalization_method):
    sp = make_snowpack(
        thickness=[1000],
        microstructure_model=microstructure_model,
        density=280,
        temperature=265,
        radius=0.05e-3,
        corr_length=0.05e-3,
    )
    scatt = active(10e9, 50)
    nstreams = 32

    # this setting fails when DORT  use scipy.linalg.eig
    # but this works with the shur diagonalization. Let check this:

    m = Model(
        emmodel,
        DORT,
        rtsolver_options=dict(m_max=m_max, n_max_stream=nstreams, diagonalization_method=diagonalization_method),
    )

    m.run(scatt, sp).sigmaVV()


def test_symmetrization():
    scatt = active(10e9, 50)
    sp = make_snowpack(
        thickness=[1000], microstructure_model="exponential", density=280, temperature=265, corr_length=0.05e-3
    )

    mu = np.array([0.5, 0.2, -0.5, -0.2])

    P = IBA(scatt, sp.layers[0]).ft_even_phase(mu, mu, m_max=1).compress(mode=1)

    symP = symmetrize_phase_matrix(P, m=1)

    npt.assert_allclose(P[0:6, 0:6], symP[0:6, 0:6])
    npt.assert_allclose(P[6:, 6:], symP[6:, 6:])

    npt.assert_allclose(P[6:, 0:6], symP[6:, 0:6])
    npt.assert_allclose(P[0:6, 6:], symP[0:6, 6:])

    npt.assert_allclose(P, symP)
