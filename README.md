# GLOW: Geospace Layered Overview Ward

The ward does the work for us.

## ezieView README

- Subset of the image generation routines developed for the EZIE Science Gateway,
  repackaged so as to be pip-installable.
- Gateway backend and operations-related routines (e.g., data inventory and scheduling)
  are not included in this package.
- Package and module names are subject to change, this is just a first pass.

## Was-Done list

- Check license, make sure right one is included
- Do we have an EZIE SPICE kernel dependency, and if so, how do we remove it?
  - Kernel manager added to download and cache required SPK/PCK/LSK files

## To-Do list

- Update outdated or missing docstrings
- Remove/rectify FIXME and TODO items
- Refactor for clarity
- Add command line options to shortcut `python3 -m ezieview.[module_name]` syntax
- Update this README to be more useful
- Update routines to use the new one-orbit-per-L1 format and the associated change in naming conventions
