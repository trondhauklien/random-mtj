import numpy as np
import numpy.typing as npt


def calc_torque(m: npt.NDArray, H_eff: npt.NDArray, M_s: float) -> np.floating:
    """
    Calculate the torque on a magnetic moment in an effective field.

    Parameters
    ----------
    m : numpy.ndarray
        The magnetic moment vector.
    H_eff : numpy.ndarray
        The effective magnetic field vector.
    M_s : float
        The saturation magnetization.

    Returns
    -------
    numpy.floating
        The magnitude of the torque.

    """
    return np.linalg.norm(np.cross(m, H_eff)) / M_s
