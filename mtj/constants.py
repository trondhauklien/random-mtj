"""Physical Constants"""

from scipy.constants import Boltzmann, physical_constants

GYROMAGNETIC_RATIO: float = physical_constants["electron gyromag. ratio"][0]
"""Gyromagnetic ratio 1.76e+11 (rad/s/T)"""

k_B: float = Boltzmann
"""BoltzmannConstant 1.38e-23 (J/K)"""

VACUUM_PERMEABILITY: float = physical_constants["vacuum mag. permeability"][0]
"""Vacuum permeability 1.26e-06 (H/m) or (N/A^2)"""

hbar: float = physical_constants["reduced Planck constant"][0]
"Reduced Planck constant 1.05e-34 (J s)"

e: float = physical_constants["elementary charge"][0]
"Elementary charge 1.6e-19 (C)"
