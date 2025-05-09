import numpy as np
import numpy.typing as ntp

from mtj.constants import GYROMAGNETIC_RATIO, VACUUM_PERMEABILITY


def LLG_rhs(m_vec, H_vec, gamma0, alpha):
    prefactor = -gamma0 / (1 + alpha**2)

    mxH = np.cross(m_vec, H_vec)
    mxmxH = np.cross(m_vec, mxH)
    torque_llg = prefactor * (mxH + alpha * mxmxH)

    return torque_llg


def LLG_Heun(
    m_i: ntp.NDArray[np.float64],
    H_eff: ntp.NDArray[np.float64],
    H_th: ntp.NDArray[np.float64],
    dt: float,
    alpha: float,
    gamma0: float = VACUUM_PERMEABILITY * GYROMAGNETIC_RATIO,
) -> ntp.NDArray[np.float64]:
    """Calculates the next time step magnetization

    Returns
    -------
        m_next: array_like
            The next time step magnetization as a NumPy array of floats with shape (3,).
    """

    H_tot = H_eff + H_th

    # Heun's method
    k1 = LLG_rhs(m_i, H_tot, gamma0, alpha)
    m_temp = m_i + dt * k1
    m_temp /= np.linalg.norm(m_temp)  # keep magnetization normalized

    k2 = LLG_rhs(m_temp, H_tot, gamma0, alpha)
    m_next = m_i + (dt / 2) * (k1 + k2)
    m_next /= np.linalg.norm(m_next)  # renormalize

    return m_next
