from typing import Unpack
import numpy as np
import numpy.typing as npt
import scipy.constants as cst
from mtj.types import MaterialProps


def calc_Heff(
    m: npt.NDArray[np.float64 | np.int_],
    stt_enable: bool = True,
    **kwargs: Unpack[MaterialProps],
) -> npt.NDArray:
    """Calculates the effective field

    Parameters
    ----------
        m: array_like
            Current magnetization as a Numpy array of floats with shape (3,).
        stt_enable: bool
            True if the STT contribution should be computed.
        **kwargs: MaterialProps
            Dictionary defining the properties of the material.
    Returns
    -------
        H_eff: array_like
            The effective field as a Numpy array of floats with shape (3,).
    """
    # magneto-crystalline anisotropy term
    H_mca = (
        2
        * kwargs["K_u"]
        / (cst.mu_0 * kwargs["M_s"])
        * np.dot(kwargs["u_k"], m)
        * kwargs["u_k"]
    )

    # demagnetization energy term
    H_d = -kwargs["M_s"] * (np.matmul(kwargs["N"], m))

    # STT term
    if stt_enable:
        H_STT = kwargs["a_para"] * kwargs["V"] * np.cross(m, kwargs["p"]) - kwargs[
            "a_ortho"
        ] * kwargs["p"] * (kwargs["V"] ** 2)
    else:
        H_STT = np.zeros(3, dtype=np.float64)

    return H_mca + kwargs["H_app"] + H_d + H_STT
