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

# ===== Date-related parameters =======================================================
UTC_TZ = datetime.UTC
DATE_NOW = datetime.datetime.now(UTC_TZ)
YEAR_NOW_INT = int(DATE_NOW.strftime("%Y"))
CLI_DATE_FORMAT = "%Y-%m-%d"  # For specifying date range on command line
EZIE_DATE_FORMAT = "%Y%m%d"  # Date format used in EZIE product filenames
SEC_PER_DAY = 24 * 60 * 60

# ===== Plot generation parameters ====================================================
DEFAULT_FIG_SIZE = (16, 10)  # Landscsape
ALTERNATE_FIG_SIZE = (16, 7)  # Short Landscsape
GLOBAL_FIG_SIZE = (16, 15)  # Short Landscsape
PORTRAIT_FIG_SIZE = (8.5, 11)
DFLT_RES = 100  # DPI
HIGH_RES = 300  # DPI, better suited for print or presentations

TERMINATOR_ALPHA = 0.25
TERMINATOR_COLOR = "#000044"
DARK_MODE_FILL_COLOR = "#202020"  # background INSIDE axes
DARK_MODE_FACE_COLOR = "#0F1116"  # background OUTSIDE axes
DARK_MODE_GRID_COLOR = "#606060"  # geodetic and magnetic grid color in dark mode
DARK_MODE_TEXT_COLOR = "#C0C0C0"  # geodetic and magnetic text/label color in dark mode

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
AVERAGING_WINDOW = 11

SPACECRAFT = ["EZIE-A", "EZIE-B", "EZIE-C"]
NUM_SPC = len(SPACECRAFT)
# Max orbits Per day when orbits cross midnight boundary, only 15 nominally.
# Might this increase as EZIE orbits decay? Edge cases? TBD
MAX_ORBITS = 16
SOUTH = "South"
EQUATORIAL = "Equatorial"
NORTH = "North"
REGIONS = [NORTH, EQUATORIAL, SOUTH]
# FIXME: Standardize these between various plot methods
BOUNDS = {}  # Must be [Lower, Upper]!
BOUNDS[NORTH] = [+45.0, +90.0]
BOUNDS[EQUATORIAL] = [-40.0, +40.0]
BOUNDS[SOUTH] = [-90.0, -45.0]
GEO_LAT_LOWER_LIMIT = 40.0
MAG_LAT_LOWER_LIMIT = 40.0
# BOUND_POLAR = 40.0  # degrees, absolute value
# BOUND_EQUATORIAL = 35.0  # degrees, absolute value

SAT_COL = "#FF800E"  # From colorblind-friendly palette

# Attitude plot parameters
CAR_CMP_NAM = ["X", "Y", "Z"]
LINTHRESH = 1.0e-3  # Bound for linear region near zero in SymLogNorm plots

# MEM plotting parameters
MEM_NUMBERS = [1, 2, 3, 4]  # handle switch from 0..3 and 1..4, indexing vs naming
NUM_MEM = len(MEM_NUMBERS)

MEM_SYMS = ["o", "o", "o", "o"]
# MEM_CLR = ["blue", "red", "green", "orange"]  # Default
MEM_CLR = [
    "orange",  #     MEM1 - (default "nadir")
    "limegreen",  #  MEM2 - negative angle from nadir
    "crimson",  #    MEM3 - negative angle from nadir
    "dodgerblue",  # MEM4 - positive angle from nadir
]  # Easier to differentiate than "Default" red-green-blue-orange
MEM_CB_CLR = [
    "C4",  # MEM1 - Cerulean/Blue
    "C5",  # MEM2 - Tenne (Tawny)/Orange
    "C9",  # MEM3 - Very Light Grey/Grey
    "C8",  # MEM4 - Macaroni and Cheese/Orange
]  # Uses colors from tableau-colorblind10 for accessibility.
# C0 and C1 are used elsewhere for plotting.

# FIXME: What angles and/or reference frames are desired here?
# MEM_LK_ANGL = [-48.7, -20.3, 0.0, 42.7]  # FIXME: Update to use JPL standard?
MEM_LOOK_DIRECTIONS = [
    +0.00,  # MEM1
    -26.0,  # MEM2
    -48.7,  # MEM3
    +42.7,  # MEM4
]  # JPL beam IDs, NOT SciBox beam numbering (order is reversed in SciBox)
# These are now defined WRT the "Primary Vector" in AEJ/Solstice science mode

EARTHLOOK = "EARTHLOOK"
SKYLOOK = "SKYLOOK"
OBSERVATION_MODES = [EARTHLOOK, SKYLOOK]

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
O2_CTR_FREQ_MHZ = 118.750e3

# Potential data product types/levels. Exact flight data product species still TBD.
L0A = "l0a"
L0B = "l0b"
L0 = "l0"
L1 = "l1"
L2 = "l2"
L3 = "l3"

# We'll match patterns in the order below, so:
# 1) Keep L0A and L0B _before_ L0
LEVELS = [
    L0A,
    L0B,
    L0,
    L1,
    L2,
    L3,
]

# Quantitities we (used to) have to calculate from netcdf variables
GEO_MAG = "S/C Geomagnetic"
MEM_MAG = "MEM Geomagnetic"
MEM_SZA = "MEM Solar Zenith Angle"
REFERENCE_ALTITUDE_KM = 80
