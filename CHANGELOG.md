
# Changelog
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Added

### Changed


## [v1.4]
### Added
	- Add a warning when snow salinity is greater than zero, but the permittivity formulation does not account for    salinity. This is only a warning since, in some cases, the mixing formula in the **emmodel** overrides the salinity in the material's permittivity formulation.  
	- Include stream angles in the output of DORT simulations. To access them in the results of a simulation (`res`), use: `res.other_data['stream_angles']`

	- Introduce a new mode for computing stream angles. This method is more uniform than Gaussian quadrature and, while less efficient, enables calculations at very high grazing incidence angles (close to 89°). Use this mode when such extreme angles are required.

	- Add make_rtsolver() and make_emmodel() to improve the readability of setting options in make_model. Refer to the core/model.py documentation for details. get_emmodel() is depreciated and the old make_emmodel() function is renamed make_emmodel_instance() but is not recommended to be used. Adapt the code to use the new make_emmodel function.

### Changed
	- depreciate Python 3.8 and lower, and add warning for future depreciation of Python 3.9. While 3.9 has a scheduled end of life for Oct 2025, type hinting is going to be implemented in SMRT soon, and the syntax provided by 3.10 is better than 3.9.


## [v1.3]
### Added

	- make_transparent_volume: allow creation of an empty medium. Useful for bare soil and open ocean calculations (no volume, with only a substrate).
	- integrate a feature-full atmosphere model (see pyrtlib_atmosphere.py) based on the open source pyrtlib package by Larosa et al. 2024 (GMD). Expect bugs at this initial stage of development.
	- add optical_depth and single_scattering_albedo methods in the result of the simulations, for convenience.

### Changed

	- The atmosphere class has been changed and is not backward compatible: renaming of the arguments and functions in to tb_up, tb_down and transmittance, and adding a run method, tb_up, tb_down and transmittance are now properties)
	- automatically remove zero-thickness layers and deal with a zero-layer snowpack by creating one zero-thickness homogenous transparent layer to fake a completely transparent volume in the rtsolver.
	- change 273 to FREEZING_POINT for some permittivity functions
	- add checks the the temperature is above or below the freezing point for water and ice permittivity formulations repsectively


## [v1.2.4]
### Added
	- implement another diagonalisation method in DORT to avoid numerical instabilities, especially in active mode. It can be activated with rtsolver_options=dict(diagonalization_method="shur_forcedtriu") in make_model. If good results are reported, this may become the default option as it seems as fast as the direct, origianl, eigenvalue solver.

### Changed
none


## [v1.2.3]
### Added

	- implement a new diagonalisation method in DORT to avoid numerical instabilities, especially in active mode. It can be activated with rtsolver_options=dict(diagonalization_method="shur") in make_model.
	- a new version of IBA is implemented for testing only. It uses the Maxwell Garnett formulation for the effective permittivity. The default IBA in iba.py is still the recommended version for normal calculations.
	- add a "snell_angles" convenience function
	- in IBA add warning for fractional volume > 0.5. Also add an argument to enable auto-model-inversion.

### Changed

	- "nonscattering" emmodel now uses polder van santen to compute the effective permittivity instead of a linear   mixing formula


## [v1.1.0]
### Added
	- Strong Expansion theory is now implemented in SMRT (see Picard et al. 2022 TC)
	- volumetric_liquid_water is a new (and recommended) way to set the amount of water in a layer using make_snowpack.
	- the 'emmodel' argument in make_model can now be a dict mapping different emmodels for each sort of layer medium. Useful for snow + sea-ice for instance when the emmodels must be different for snow and ice.
	- a new function in make_medium.py to create a water body (lake or open ocean): from smrt import make_water_body


### Changed
	- Fresnel coefficients for the reflection and transmission on flat interface are now calculated with a more rigorous equation for (very) lossly materials. This could affect some simulations with water but the effect is most > 60° incidence angle.
