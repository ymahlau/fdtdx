
# Unpublished
## Added
- added new description for the placeholder attributes (@renaissancenerd)
- added plot_field_slice function which creates 2x3 images (with colorbar) for given E and H fields (@renaissancenerd)

## Changed
- fixed the detector plotting bug (@renaissancenerd)

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


