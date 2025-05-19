from typing import Unpack
import numpy as np
import numpy.typing as npt
import scipy.constants as cst
import mtj.constants as m_cst
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

def calc_e(
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
        e: float
            Total energy normalized per unit volume.
    """
    e_app = -cst.mu_0 * kwargs["M_s"] * np.dot(kwargs["H_app"],m)
    N_d = np.diagonal(kwargs["N"])
    e_dem = 0.5*cst.mu_0*(kwargs["M_s"]**2)*(N_d[0]*(m[0]**2)+
                                             N_d[1]*(m[1]**2)+
                                             N_d[2]*(m[2]**2))
    e_k = kwargs["K_u"]*(1-np.dot(m,kwargs["u_k"])**2)

    e_stt = 0
    # From the STT torque: the energy contribution is defined by analogy with an externally
    # applied magnetic field.
    #if stt_enable:
    #    e_stt = -cst.mu_0*kwargs["M_s"]*np.dot(m,kwargs["a_para"]*kwargs["V"]*np.cross(m,p)-
    #                                           kwargs["a_ortho"]*(kwargs["V"]**2)*kwargs["p"]))

    return e_app + e_dem + e_k + e_stt
