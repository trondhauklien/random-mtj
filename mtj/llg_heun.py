import numpy as np
import numpy.typing as ntp

from mtj.constants import GYROMAGNETIC_RATIO


def LLG_rhs(p_vec, m_vec, H_vec, gamma0, alpha, a_perp, a_par):
    prefactor = -gamma0 / (1 + alpha**2)

    mxH = np.cross(m_vec, H_vec)
    mxmxH = np.cross(m_vec, mxH)
    torque_llg = prefactor * (mxH + alpha * mxmxH)

    # STT terms
    mxp = np.cross(m_vec, p_vec)
    mxmxp = np.cross(m_vec, mxp)
    stt_term = -gamma0 * a_par * mxmxp + gamma0 * a_perp * mxp

    return torque_llg + stt_term


def LLG_Heun(
    m_i: ntp.NDArray[np.float64],
    p_i: ntp.NDArray[np.float64],
    H_eff: ntp.NDArray[np.float64],
    H_th: ntp.NDArray[np.float64],
    dt: float,
    alpha: float,
    a_perp: float,
    a_par: float,
    # TODO Verify this value
    gamma0: float = 2.21e5,  # gyromagnetic ratio in (m/As)
) -> ntp.NDArray[np.float64]:
    """Calculates the next time step magnetization

    Returns
    -------
        m_next: array_like
            The next time step magnetization as a NumPy array of floats with shape (3,).
    """

    # Normalize input vectors in place
    m_i = m_i / np.linalg.norm(m_i)
    p_i = p_i / np.linalg.norm(p_i)

    H_tot = H_eff + H_th

    # Heun's method
    k1 = LLG_rhs(p_i, m_i, H_tot, gamma0, alpha, a_perp, a_par)
    m_temp = m_i + dt * k1
    m_temp /= np.linalg.norm(m_temp)  # keep magnetization normalized

    k2 = LLG_rhs(p_i, m_temp, H_tot, gamma0, alpha, a_perp, a_par)
    m_next = m_i + (dt / 2) * (k1 + k2)
    m_next /= np.linalg.norm(m_next)  # renormalize

    return m_next
