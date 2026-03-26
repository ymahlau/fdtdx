
# Unpublished
## Added
- added progressbar for FDTD simulations and runs (@renaissancenerd)
- added typechecking to the CI/CD pipeline (@elyes298)
- Implemented PEC and PMC boundary conditions with field enforcement (@bruxillensis, #264)
- Implemented Bloch boundary conditions with configurable phase vector for band structure simulations (@bruxillensis, #264)
- `PeriodicBoundary` is now an alias for `BlochBoundary` with zero Bloch vector, eliminating redundant code (@bruxillensis, #264)
- Automated complex field detection via `SimulationConfig.use_complex_fields` when non-zero Bloch vector is present (@bruxillensis, #264)
- Added 1D photonic crystal band-structure example demonstrating Bloch boundary conditions (@bruxillensis, #264)
- Reorganized test suite into unit, integration, and simulation tiers with pytest markers for selective execution (@bruxillensis, #257)
- Significantly increased code coverage across all source files (@bruxillensis, #257)
## Changed
- Bug fixes in `curl.py`, `diffractive.py`, `mode.py`, and `continuous.py` (@bruxillensis, #257)
## Removed


# v0.6.0 (Feb 13 2026)
## Added
- added new description for the placeholder attributes (@renaissancenerd)
- added plot_field_slice function which creates 2x3 images (with colorbar) for given E and H fields (@renaissancenerd)
- refactored Color with comprehensive colors from the XKCD color survey  (@renaissancenerd)
- added option to export to VTI file for visualization with external tools (e.g., ParaView) (@l-berg)
- Implemented support for anisotropic materials (diagonal and fully tensorial) (@rachsmith)
- Added Component example notebook showcasing uniform plane source (@Shardy2907)
- Mode source and mode detector now expose the effective index (@l-berg)

## Changed
- fixed the detector plotting bug (#220) (@renaissancenerd)
- refactored GaussianPulseProfile by adding fdtdx.WaveCharacter (#90) (@renaissancenerd)
- Improved Documentation (@ymahlau)

## Removed


# v0.5.0 (Dec 02 2025)
## Added
- separate repository for notebooks rendered in docs: https://github.com/ymahlau/fdtdx-notebooks
- dependabot / codecov integration for better monitoring of repository
- A lot of test cases (@renaissancenerd)
- Json import / export of simulation objects. This will be necessary and helpful for the ongoing GUI development.
- added plot_material function. This function plots the material distribution at a specific slice in the simulation. (@renaissancenerd)

## Changed
- moved docs to readthedocs using sphinx instead of mkdocs. This looks much nicer now.
- fixed eta0 scaling for lossy material (@rachsmith1)
- PML boundary layer refactoring (@rachsmith1)
- refactored plot_setup. (@renaissancenerd)
- refactored place_objects API. This is a breaking change since now a list of simulation objects needs to be included as an attribute to the function.

## Removed
- lossy material example. Going forward, examples will be included in the notebooks repository (https://github.com/ymahlau/fdtdx-notebooks) and rendered in the documentation.


