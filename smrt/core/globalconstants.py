# coding: utf-8

""" Global constants used throughout the model are defined here and imported as needed.
    The constants are:

    =====================   ===============================   =================================================
    Parameter               Description                       Value
    =====================   ===============================   =================================================
    DENSITY_OF_ICE          Density of pure ice at 273.15K    916.7 kg m :sup:`-3`
    FREEZING_POINT          Freezing point of pure water      273.15 K
    C_SPEED                 Speed of light in a vacuum        2.99792458 x 10 :sup:`8` ms :sup:`-1`
    PERMITTIVITY_OF_AIR     Relative permittivity of air      1
    =====================   ===============================   =================================================

    **Usage example:**

        ::

            from smrt.core.globalconstants import DENSITY_OF_ICE
"""

import numpy as np

DENSITY_OF_ICE = 916.7
DENSITY_OF_WATER = 1000
FREEZING_POINT = 273.15
PERMITTIVITY_OF_AIR = 1.
C_SPEED = 299792458.
PERMITTIVITY_OF_FREE_SPACE = 1 / (4e-7 * np.pi * C_SPEED ** 2)
EARTH_RADIUS = 6371000.

GHz = 1e9    # G in fact
cm = 1e-2    # centimeter
mm = 1e-3    # millimeter
micron = 1e-6 # micron
PSU = 1e-3   # g/kg -> kg/kg
