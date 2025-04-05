import numpy as np
import numpy.typing as ntp


def calc_Heff(
    m: ntp.NDArray[np.float64],
    # More parameters goes here
) -> ntp.NDArray[np.float64]:
    """Calculates the effective field

    Parameters
    ----------
        m: array_like
            Current magnetization as a Numpy array of floats with shape (3,).

    Returns
    -------
        H_eff: array_like
            The effective field as a Numpy array of floats with shape (3,).
    """

    # The implementation goes here

    return np.array([0,0,0])  # TODO Replace this after implementing function
