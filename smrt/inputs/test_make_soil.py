import numpy as np
import pytest

from smrt.core.error import SMRTError
from smrt.inputs.make_soil import make_soil_column, make_soil_substrate
from smrt.substrate.flat import Flat


@pytest.fixture
def soil_parameters():
    return dict(
        thickness=np.array([0.50, 1]),  # 50 cm thick layers
        moisture=np.array([0.05, 0.10]),
        sand=np.array([0.11, 0.15]),
        clay=np.array([0.18, 0.15]),
        drymatter=np.array([1300, 1100]),
        temperature=np.array([293, 283]),
    )


def test_make_soil():
    substrate = make_soil_substrate(
        "flat",
        "soil_permittivity_dobson85_peplinski95",
        temperature=293,
        moisture=0.05,
        sand=0.11,
        clay=0.18,
    )

    assert isinstance(substrate, Flat)


def test_make_soil_default_permittivity():
    substrate = make_soil_substrate("flat")

    assert substrate.permittivity_model.__name__ == "soil_permittivity_dobson85_peplinski95"


def test_make_soil_column(soil_parameters):
    # Create a soil column

    soil_column = make_soil_column(**soil_parameters)

    assert soil_column.layers[0].thickness == 0.50
    assert soil_column.layers[1].permittivity_model[0].__name__ == "soil_permittivity_dobson85_peplinski95"
    assert soil_column.layers[0].microstructure_model.__name__ == "Homogeneous"
    assert soil_column.layers[0].frac_volume == 0


def test_make_soil_column_with_substrate(soil_parameters):
    # Create a soil column with a substrate

    soil_column = make_soil_column(
        **soil_parameters,
        add_soil_substrate=True,
    )

    assert len(soil_column.layers) == 3
    assert isinstance(soil_column.layers[-1].interface, Flat)


def test_make_soil_column_all_zero_thickness(soil_parameters):
    soil_parameters["thickness"] = np.array([0.0, 0.0])
    soil_column = make_soil_column(**soil_parameters)

    assert len(soil_column.layers) == 1
    assert soil_column.layers[0].thickness == 0
    assert soil_column.layers[0].microstructure_model.__name__ == "Homogeneous"


def test_make_soil_column_surface_and_list_interface_raises(soil_parameters):
    with pytest.raises(SMRTError):
        make_soil_column(
            **soil_parameters,
            surface="flat",
            interface=["flat", "flat"],
        )
