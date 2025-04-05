"""Physical Constants"""

from scipy.constants import Boltzmann, physical_constants

GYROMAGNETIC_RATIO: float = physical_constants["electron gyromag. ratio"][0]
"""Gyromagnetic ratio (rad/s/T)"""

k_B: float = Boltzmann
"""BoltzmannConstant (J/K)"""

VACUUM_PERMEABILITY: float = physical_constants["vacuum mag. permeability"][0]
"""Vacuum permeability (H/m) or (N/A^2)"""
