import numpy as np
import numpy.typing as ntp

from mtj.constants import GYROMAGNETIC_RATIO, VACUUM_PERMEABILITY, k_B


def compute_thermal_field(
    alpha,
    T,
    M_s,
    V,
    dt,
    k_B=k_B,
    mu0=VACUUM_PERMEABILITY,
    gamma0=GYROMAGNETIC_RATIO * VACUUM_PERMEABILITY,
) -> ntp.NDArray[np.float64]:
    """
    Compute a single 3D vector of thermal magnetic field H_th
    with zero mean and variance based on fluctuation-dissipation theorem.

    Parameters
    ----------
        alpha: float
            Damping factor (unitless).

        T: float
            Temperature (K).

        M_s: float
            Saturation magnetization module (A/m).

        V: float
            Volume (m^3).

        dt: float
            Time step (s).

        k_B: float, optional
            Boltzmann Constant (J/K).

        mu0: float, optional
            Vacuum permeability (N/A^2).

        gamma0: float, optional
            Gyromagnetic ratio: gamma0=gamma*mu_0 (m/(A·s)).

    Returns
    -------
        H_th: array_like
            The thermal magnetic field as a Numpy array of floats with shape (3,).
    """

    # Variance constant D
    D = (2 * alpha * k_B * T) / (mu0 * M_s * V * gamma0)

    # Standard deviation for each component of the field
    std_dev = np.sqrt(D / dt)

    # Generate 3 normally distributed values for Hx, Hy, Hz
    H_th = np.random.normal(loc=0.0, scale=std_dev, size=3)

    return H_th
