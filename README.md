# ezieView

## About

- The ezieView python package is a subset of the image generation routines developed for
  the EZIE Science Gateway, repackaged so as to be pip-installable.

- A SPICE kernel manager utility has been included to download and cache the SPK/PCK/LSK
  files required to compute the positions of the Sun and Earth so as to properly orient
  certain subplots.

- Make images by running any of the following or alternatively using a shell script like
  the one shown below.

## CLI Usage

```bash
. .venv/bin/activate

python3 -m ezieview.update_coverage_plots [CLI options]
python3 -m ezieview.update_orbit_plots [CLI options]
```

... or use the alternate script entry points provided in the package installation:

`view_coverage [CLI options]`  

`view_orbits [CLI options]`[^1]

Run the scripts with the `--help` flag to display interactive help for each utility,
e.g.,

```bash
python3 -m ezieview.update_orbit_plots --help
```

***

[^1]: `view_coverage` and `view_orbits` have been defined as alternate command line entry
points for `ezieview.update_coverage_plots` and `ezievew.update_orbit_plots`,
respectively.

## Shell script interface

- The `make_plots.sh` shell script will generate all images for a given date range, as
  defined by the EZ_BGN_DATE and EZ_END_DATE variable settings, which may be edited in
  the script.

```bash
#!/usr/bin/env bash
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
EZ_L1_DATA="${EZ_DATA_ROOT}/l1/orbit"
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
# 5) B field and Current retrievals on same figure
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

python3 -m ezieview.update_orbit_plots \
    -d0 "${EZ_BGN_DATE}" \
    -d1 "${EZ_END_DATE}" \
    -fd "${EZ_L2_DATA}" \
    -pd "${EZ_L2_PLOT}" \
    -over "${EZ_OVERWRITE}" -merged

```
