import numpy as np
import numpy.typing as npt


def init_m(m0: npt.NDArray[np.float64], Nt: int) -> npt.NDArray[np.float64]:
    """Initializes a magnetization matrix

    Parameters
    ----------
        m0: array_like
            Initial magnetization
        Nt: int
            Total number of timesteps

    Returns
    -------
        m: array_like
            A NumPy array of floats with shape (Nt, 3).
    """
    m = np.zeros((Nt, len(m0)), dtype=np.float64)
    m[0] = m0
    return m
