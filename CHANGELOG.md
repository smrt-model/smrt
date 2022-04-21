
# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Added
	- volumetric_liquid_water is a new (and recommended) way to set the amount of water in a layer using make_snowpack.

	- the 'emmodel' argument in make_model can now be a dict mapping a different emmodel for each sort of layer medium. Useful for snow + sea-ice for instance when the emmodels must be different for snow and ice.
