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
- Update routines to use the new one-orbit-per-L1 format and the associated change in
  naming conventions
- Add option to define option values with *implicit* evars (set in environment) rather
  than *explicit* evars (passed in with the option flag, e.g., -d0 $START_DATE)
- Add a multiprocessing pool option to speed up plot generation?

## Build instructions (temporary notes to self)

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

## Install GLOW/ezieView

```bash
mkdir ezvtest
cd ezvtest
uv venv -p 3.12
UV_CACHE_DIR=/project/ezie/.cache/uv uv pip install ../glow
. .venv/bin/activate
```

- Make images by running any of the following:

```bash
.venv/bin/make_ezie_images.sh

python3 -m ezieview.update_coverage_plots [CLI options]
... or 
glow_daily [CLI options]

python3 -m ezieview.update_orbit_plots [CLI options]
... or 
glow_orbit [CLI options]
```

## Shell script interface

- The following shell script will generate all images for a given date range, as defined
  by START_DATE and STOP_DATE below.

```bash
#!/bin/bash
# =====================================================================================
# Generate a set of images from the relevant EZIE data products.
#   Select run dates and set desired value of EZ_OVERWRITE.
#   EZ_OVERWRITE = True  ==> Generate new images, replacing any that already exist
#   EZ_OVERWRITE = False ==> Generate only those images that do NOT already exist
# =====================================================================================

# EZ_OVERWRITE=True
EZ_OVERWRITE=False
EZ_BGN_DATE="2026-02-22"
EZ_END_DATE="2026-02-22"

# =====================================================================================
# Specify product source directories to be processed & corresponding output locations
# =====================================================================================

EZ_DATA_ROOT="/project/ezie/data"
EZ_L1_DATA="${EZ_DATA_ROOT}/l1"
EZ_L2_DATA="${EZ_DATA_ROOT}/l2"
EZ_L3_DATA="${EZ_DATA_ROOT}/l3"

EZ_PLOT_ROOT="./outputs"
EZ_DAILY_PLOT="${EZ_PLOT_ROOT}/daily-summary"
EZ_L1_PLOT="${EZ_PLOT_ROOT}/single-orbit"
EZ_L2_PLOT="${EZ_PLOT_ROOT}/single-orbit"
EZ_L3_PLOT="${EZ_PLOT_ROOT}/single-orbit"

# =====================================================================================
# Record configuration for this run to a (daily) log file (optional).
# =====================================================================================

if [ ! -d "./logs" ]; then
    mkdir "./logs"
fi
LOGFILE="./logs/$(date +"%Y%m%d")_config.log"
touch "$LOGFILE"
{
echo "==================================================================="
echo "Current environment configuration as of $(date)"
echo "==================================================================="
set | grep EZ_ | grep "DATE" | sort
set | grep EZ_ | grep "PTRN" | sort
set | grep EZ_ | grep "DATA" | sort
set | grep EZ_ | grep "PLOT" | sort
set | grep EZ_ | egrep -v "DATE|PTRN|DATA|PLOT"  | sort
echo "==================================================================="
} >> "$LOGFILE" 

# =====================================================================================
# Generate image products. Comment in/out product level/type commands as desired.
# 1) All day orbit/SV coverage --  Mlat/MLT/SZA 
# 2) Ancillary, geolocation, TA, TB
# 3) B field retrieval, dBd[own] only ATM
# 4) Current retrieval, based on dBd only ATM
# =====================================================================================

python3 -m ezieview.update_coverage_plots \
    -d0 "${EZ_BGN_DATE}" \
    -d1 "${EZ_END_DATE}" \
    -fd "${EZ_L1_DATA}" \
    -pd "${EZ_DAILY_PLOT}" \
    -over "${EZ_OVERWRITE}"

python3 -m ezieview.update_orbit_plots \
    -d0 "${EZ_BGN_DATE}" \
    -d1 "${EZ_END_DATE}" \
    -fd "${EZ_L1_DATA}" \
    -pd "${EZ_L1_PLOT}" \
    -over "${EZ_OVERWRITE}"

python3 -m ezieview.update_orbit_plots \
    -d0 "${EZ_BGN_DATE}" \
    -d1 "${EZ_END_DATE}" \
    -fd "${EZ_L2_DATA}" \
    -pd "${EZ_L2_PLOT}" \
    -over "${EZ_OVERWRITE}"

python3 -m ezieview.update_orbit_plots \
    -d0 "${EZ_BGN_DATE}" \
    -d1 "${EZ_END_DATE}" \
    -fd "${EZ_L3_DATA}" \
    -pd "${EZ_L3_PLOT}" \
    -over "${EZ_OVERWRITE}"

  ```
