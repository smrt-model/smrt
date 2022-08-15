
# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Added
	- volumetric_liquid_water is a new (and recommended) way to set the amount of water in a layer using make_snowpack.

	- the 'emmodel' argument in make_model can now be a dict mapping of different emmodels for each sort of layer medium. Useful for snow + sea-ice for instance when the emmodels must be different for snow and ice.

	- a new function in make_medium.py to create a water body (lake or open ocean): from smrt import make_water_body  


### Changed
	- Fresnel coefficients for the reflection and transmission on flat interface are now calculated with a more rigorous equation for (very) lossly materials. This could affect some simulations but with very little effect 60°.

