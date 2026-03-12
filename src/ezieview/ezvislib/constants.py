AVOGADRO = 6.02214076e23
"""
Number of particles per mole
"""

BOHR_MAGNETON = 9.2740100783e-24
"""
Bohr magneton in J/T
"""

BOLTZMANN_K = 1.380649e-23
"""
Boltzmann constant k in joules/kelvin (kg-m^2/s^2/K). To convert to g-cm^2/s^2/K
multiply by 1e7.  

If this value is changed, please change the value in lib_ezieAsPy_c/EZIEConsts.h!
"""

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

LOSCHMIDT = 2.686780111e19
"""
Number per cubic CENTIMETER, 1e-6 times value per cubic meter quoted below!

Loschmidt constant = Number of particles (atoms or molecules) of an ideal gas per volume
(the number density), usually quoted at standard temperature and pressure. The 2018
CODATA recommended value is 2.686780111x10^25 per cubic metre at 0 degrees C and 1 atm. 
"""

AMAGAT = LOSCHMIDT
"""
Number of ideal gas molecules per unit volume at 1 atm (101.325 kPa) and 0 degrees C
(273.15 K). Being a measure of number density, the Loschmidt constant n_0 is used to
define the amagat, a practical unit of number density for gases and other substances:

    1 amagat = n_0 = 2.686780111x10^25 m^-3,

such that the Loschmidt constant is exactly 1 amagat.
"""

MU_E_MU_B_RATIO = 1.001159  # Magnitude, unsigned
"""
Electron magnetic moment to Bohr magneton ratio, mu_e/mu_B 
Numerical value	              -1.001 159 652 180 46
Standard uncertainty	       0.000 000 000 000 18
Relative standard uncertainty  1.8 x 10-13
Concise form	              -1.001 159 652 180 46(18)
Source: https://physics.nist.gov/cgi-bin/cuu/Value?muemsmub
"""

O2_MOLAR_MASS = 31.999
"""
O2 molar mass in g/mole
"""

PLANCK_CONSTANT = 6.62607015e-34
"""
Planck constant in J s

If this value is changed, please change the value in lib_ezieAsPy_c/EZIEConsts.h!
"""

SLS_PRESSURE_MBAR = 1013.25
"""
Pressure in millibars (hectoPascals, 10^2 N/m^2) for Sea-Level Standard (SLS) conditions
"""

WAVENUMBER_TO_KHZ = C * 100 / 1e3
"""
Conversion from wavenumber in cm^{-1} to kilohertz
"""

WAVENUMBER_TO_MHZ = C * 100 / 1e6
"""
Conversion from wavenumber in cm^{-1} to megahertz
"""

ZERO_DEGREES_C = 273.15
"""
Kelvin equivalent to zero degrees celsius, used as a baseline for some atmospheric
calculations.
"""

# TODO: put this in a config file
t_doppler = 200
"""
This is not a constant, but placed here until we put it in a config file"""
