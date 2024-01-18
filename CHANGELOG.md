
# Changelog
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Added
### Changed


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

