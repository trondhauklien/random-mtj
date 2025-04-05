import numpy as np
import numpy.typing as ntp

from mtj.constants import GYROMAGNETIC_RATIO, VACUUM_PERMEABILITY, k_B


def compute_thermal_field(
    alpha, T, Ms, V, dt, k_B=k_B, mu_0=VACUUM_PERMEABILITY, gamma=GYROMAGNETIC_RATIO
) -> ntp.NDArray[np.float64]:
    """
    Compute a single 3D vector of thermal magnetic field H_th
    with zero mean and variance based on fluctuation-dissipation theorem.

    Parameters
    ----------
        temperature: float
            in Kelvin.

    Returns
    -------
        H_th: array_like
            The thermal magnetic field as a Numpy array of floats with shape (3,).
    """

    # Variance constant D
    D = (2 * alpha * k_B * T) / (mu_0 * Ms * V * gamma)

    # Standard deviation for each component of the field
    std_dev = np.sqrt(D / dt)

    # Generate 3 normally distributed values for Hx, Hy, Hz
    H_th = np.random.normal(loc=0.0, scale=std_dev, size=3)

    return H_th
