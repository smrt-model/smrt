# coding: utf-8


"""
Implement the radar_calibration_sphere interface boundary for the bottom layer (substrate).

"""

# local import
from smrt.interface.radar_calibration_sphere import RadarCalibrationSphere as iRadarCalibrationSphere
from smrt.core.interface import substrate_from_interface

# autogenerate from interface.radar_calibration_sphere
@substrate_from_interface(iRadarCalibrationSphere)
class RadarCalibrationSphere:
    pass


