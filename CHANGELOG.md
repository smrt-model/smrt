
# Changelog
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Added

### Changed


## [v1.1.0]
### Added
	- Strong Expansion theory is now implemented in SMRT (see Picard et al. 2022 TC)

	- volumetric_liquid_water is a new (and recommended) way to set the amount of water in a layer using make_snowpack.

	- the 'emmodel' argument in make_model can now be a dict mapping different emmodels for each sort of layer medium. Useful for snow + sea-ice for instance when the emmodels must be different for snow and ice.

	- a new function in make_medium.py to create a water body (lake or open ocean): from smrt import make_water_body  


### Changed
	- Fresnel coefficients for the reflection and transmission on flat interface are now calculated with a more rigorous equation for (very) lossly materials. This could affect some simulations with water but the effect is most > 60° incidence angle.

