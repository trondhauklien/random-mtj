import numpy as np
import numpy.typing as ntp


def calc_Hfl(
    temperature: float,  # Kelvin
    # More parameters goes here
) -> ntp.NDArray[np.float64]:
    """Calculates the effective field

    Parameters
    ----------
        temperature: float
            in Kelvin.

    Returns
    -------
        H_fl: array_like
            The thermal fluctuations field as a Numpy array of floats with shape (3,).
    """

    # The implementation goes here

    return np.array([0, 0, 0])  # TODO Replace this after implementing function
