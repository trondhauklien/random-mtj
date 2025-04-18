import numpy as np
import numpy.typing as ntp
import scipy.constants as cst


def calc_Heff(
    m: ntp.NDArray[np.float64],
    K_u: float,
    M_s: float,
    u_k: ntp.NDArray[np.float64],
    p: ntp.NDArray[np.float64],
    a_para: float,
    a_orth: float,
    V: float,
    H_app: ntp.NDArray[np.float64],
    N: ntp.NDArray[np.float64],
    stt_enable: bool = True,
) -> ntp.NDArray[np.float64]:
    """Calculates the effective field

    Parameters
    ----------
        m: array_like
            Current magnetization as a Numpy array of floats with shape (3,).
        K_u: float
            First order crystal anisotropy constant (J/m^3).
        M_s: float
            Saturation magnetization module (A/m).
        u_k: array_like
            Direction of easy axis as a Numpy array of floats with shape (3,).
        p: array_like
            Spin polarization unit vector as a Numpy array of floats with shape (3,).
        a_para: float
            Parallel STT coefficient.
        a_ortho: float
            Orthogonal STT coefficient.
        V: float
            Applied voltage (V).
        H_app: array_like
            Externally applied magnetic field as a Numpy array of floats with shape (3,).
        N: array_like
            Demagnetization tensor as a Numpy array of floats with shape (3,3).
        stt_enable: bool
            True if the STT contribution should be computed.

    Returns
    -------
        H_eff: array_like
            The effective field as a Numpy array of floats with shape (3,).
    """
    # magneto-crystalline anisotropy term
    H_mca = 2*K_u/(cst.mu_0*M_s)*np.dot(u_k, m)*u_k

    # demagnetization energy term
    H_d = -M_s*(np.matmul(N, m))

    # STT term
    if stt_enable:
        H_STT = a_para*V*np.cross(m,p) - a_orth*p*(V**2)
    else:
        H_STT = np.zeros(3, dtype=np.float64)

    return H_mca + H_app + H_d + H_STT
