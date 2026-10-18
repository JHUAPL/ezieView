"""
Shared constants for the EZIE science gateway plotting codes.

Defines physical constants, plot sizing/resolution/color parameters, region
definitions and bounds, and spacecraft and MEM identifiers used by both the
gateway plotting routines (gw_*.py) and the update_*_plots.py scripts.
"""

import datetime

import cartopy.crs as ccrs

# ===== Physical constants ============================================================
C = 2.99792458e8
"""
Speed of light in m/s
"""

EARTH_FLATTENING = 1 / 298.257223563
"""
WGS84 value for flattening of the Earth.  This is (a-b)/a where a is the equatorial and
b is the polar radius.
"""

EARTH_RADIUS_EQUATORIAL = 6378.137
"""
WGS84 value for equatorial radius of the Earth, in km.
"""

EARTH_RADIUS_POLAR = 6356.752314245
"""
WGS84 value for polar radius of the Earth, in km.
"""
# ===== Model/retrieval parameters =======================================================
REFERENCE_ALTITUDE_KM = 80
"""
Altitude in km at which MEM footprints are geolocated
"""

O2_CTR_FREQ_MHZ = 118.750e3
"""
[Unshifted rest] Center frequency of observed O2 transition in MHz
"""

# ===== Date-related parameters =======================================================
UTC_TZ = datetime.UTC
CLI_DATE_FORMAT = "%Y-%m-%d"  # For specifying date range on command line
EZIE_DATE_FORMAT = "%Y%m%d"  # Date format used in EZIE product filenames

# ===== Plot generation parameters ====================================================
DEFAULT_FIG_SIZE = (16, 10)  # Landscape
ALTERNATE_FIG_SIZE = (16, 7)  # Short Landscape
DFLT_RES = 100  # DPI

TERMINATOR_ALPHA = 0.25
TERMINATOR_COLOR = "#000044"
DARK_MODE_FILL_COLOR = "#202020"  # background INSIDE axes
# DARK_MODE_FACE_COLOR = "#0F1116"  # background OUTSIDE axes
DARK_MODE_FACE_COLOR = "#282828"  # background OUTSIDE axes
DARK_MODE_GRID_COLOR = "#606060"  # geodetic and magnetic grid color in dark mode
DARK_MODE_TEXT_COLOR = "#A0A0A0"  # geodetic and magnetic text/label color in dark mode

# Data collection and MEM observation plots - mapped
SCIENCE_COLOR = "purple"
SCIENCE_LABEL = "Science"
SPCLOOK_COLOR = "orange"
SPCLOOK_LABEL = "Space-look"
BGN_COLOR = "green"
END_COLOR = "red"
EZIE_COLOR = "black"
EZIE_LABEL = "Spacecraft"
MAGLAT, MLT, SZA = "Magnetic Latitude", "Magnetic Local Time", "Solar Zenith Angle"
COVERAGE_TYPES = [MAGLAT, MLT, SZA]
COVERAGE_UNITS = ["degrees", "hours", "degrees"]

# Mapping parameters
DATA_TRANSFORM = ccrs.PlateCarree()

# Was 9--used to smooth noisy values for plotting. Current test data has no added noise.
AVERAGING_WINDOW = 5

SPACECRAFT = ["EZIE-A", "EZIE-B", "EZIE-C"]
NUM_SPC = len(SPACECRAFT)
# Max orbits Per day when orbits cross midnight boundary, only 15 nominally.
# Might this increase as EZIE orbits decay? Edge cases? TBD
MAX_ORBITS = 16
SOUTH = "South"
EQUATORIAL = "Equatorial"
NORTH = "North"
REGIONS = [NORTH, EQUATORIAL, SOUTH]
# TODO: Standardize these between various plot methods and remove redundancies
BOUNDS = {  # Must be [Lower, Upper]!
    NORTH: [+45.0, +90.0],
    EQUATORIAL: [-40.0, +40.0],
    SOUTH: [-90.0, -45.0],
}
GEO_LAT_LOWER_LIMIT = 40.0
MAG_LAT_LOWER_LIMIT = 40.0

SAT_COL = "#FF800E"  # From colorblind-friendly palette

# MEM plotting parameters
MEM_NUMBERS = [1, 2, 3, 4]  # handle switch from 0..3 and 1..4, indexing vs naming
NUM_MEM = len(MEM_NUMBERS)

MEM_SYMS = ["o", "o", "o", "o"]
MEM_CLR = [
    "orange",  #     MEM1 - (default "nadir")
    "limegreen",  #  MEM2 - negative angle from nadir
    "crimson",  #    MEM3 - negative angle from nadir
    "dodgerblue",  # MEM4 - positive angle from nadir
]  # Easier to differentiate than "Default" red-green-blue-orange

# TODO: What angles and/or reference frames preferred here? Update to use JPL standard?
MEM_LOOK_DIRECTIONS = [
    +0.00,  # MEM1
    -26.0,  # MEM2
    -48.7,  # MEM3
    +42.7,  # MEM4
]  # JPL beam IDs, NOT SciBox beam numbering (order is reversed in SciBox)
# These are now defined WRT the "Primary Vector" in AEJ/Solstice science mode

EARTHLOOK = "EARTHLOOK"
SKYLOOK = "SKYLOOK"

# Retrieved and background B field plotting parameters
BN = "B$_N$"
BE = "B$_E$"
BD = "B$_D$"
BT = "B$_{TOT}$"
FLD_CMP = [BN, BE, BD, BT]
FLD_CLR = ["blue", "red", "limegreen", "orange"]
NUM_FLD = len(FLD_CMP)
TOT_MOD = "Total"
DBS_MOD = "dBs"
FLD_MODES = [TOT_MOD, DBS_MOD]
