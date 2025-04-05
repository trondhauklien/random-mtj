import numpy as np
import numpy.typing as ntp

from mtj.constants import GYROMAGNETIC_RATIO, k_B


def LLG_Heun(
    m: ntp.NDArray[np.float64],
    H_eff: ntp.NDArray[np.float64],
    # More parameters goes here
) -> ntp.NDArray[np.float64]:
    """Calculates the next time step magnetization

    Parameters
    ----------
        m: array_like
            Magnetization as a Numpy array of floats with shape (3,).
        H_eff: array_like
            The effective field

    Returns
    -------
        m_next: array_like
            The next time step magnetization as a NumPy array of floats with shape (3,).
    """

    # The implementation goes here

    return np.array([0, 0, 0])  # TODO Replace this after implementing function
