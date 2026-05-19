# ezieView

- The ezieView python package is a subset of the image generation routines developed for
  the EZIE Science Gateway, repackaged so as to be pip-installable.

- A SPICE kernel manager utility has been included to download and cache the SPK/PCK/LSK
  files required to compute the positions of the Sun and Earth so as to properly orient
  certain subplots.

## To-Do list (To be removed for release)

- Routine gateway backend and operations-related routines (e.g., data inventory and
  scheduling) are NOT included in this package.
- Update outdated or missing docstrings
- Remove/rectify FIXME and TODO items
- Refactor for clarity
- Update this README to be more useful
- Update routines to use the new file naming conventions that include the orbit number
  when that comes to pass.
- Add option to define option values with *implicit* evars (set in environment) rather
  than *explicit* evars (passed in with the option flag, e.g., -d0 $START_DATE)
- Add a multiprocessing pool option to speed up plot generation?

## Build/install instructions (Temporary notes to self, also to be removed for release)

- In parent directory:

```bash
uv venv -p 3.12
uv pip install build
. .venv/bin/activate
UV_CACHE_DIR=/project/ezie/.cache/uv PIP_INDEX_URL=https://pypi.org/simple python3 -m build glow
```

- Note that specifiying a non-standard location for UV_CACHE_DIR is only useful on DMZ
  systems where user HOME directory space is limited and the uv cache directory may also
  be mounted on a different disk volume than the one where the code and venv are
  installed.

- Note also that the use of setup.py/setuptools is now deprecated.

### Install ezieView

- Create the directory where you'd like to install the package and initialize your venv:

```bash
mkdir ezvtest
cd ezvtest
uv venv -p 3.12
```

- Option A, for developers: Install in "editable" mode, allowing you to modify the
  package code and see the changes immediately:

```bash
UV_CACHE_DIR=/project/ezie/.cache/uv uv pip install -e ../ezieView
```

- Option B, for regular users: Install in the normal fashion. Eventually the package
  will be available from the Artifactory (JHUAPL) or PyPi (rest of world) and will be
  installed by name rather than a local file path/wheel.

```bash
UV_CACHE_DIR=/project/ezie/.cache/uv uv pip install ../ezieView/dist/ezieview-0.0.1-py3-none-any.whl
```

- Make images by running any of the following or alternatively using a shell script like
  the one shown below.

```bash
. .venv/bin/activate

python3 -m ezieview.update_coverage_plots [CLI options]
... or 
view_coverage [CLI options]

python3 -m ezieview.update_orbit_plots [CLI options]
... or 
view_orbits [CLI options]
```

## Shell script interface

- The `make_plots.sh` shell script will generate all images for a given date range, as
  defined by the EZ_BGN_DATE and EZ_END_DATE variable settings, which may be edited in
  the script.
