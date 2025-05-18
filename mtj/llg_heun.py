from typing import Unpack

import numpy as np
import numpy.typing as ntp

from mtj.constants import GYROMAGNETIC_RATIO, VACUUM_PERMEABILITY
from mtj.calc_Heff import calc_Heff
from mtj.calc_Hth import compute_thermal_field
from mtj.types import MaterialProps


def LLG_rhs(m_vec, H_vec, gamma0, alpha):
    prefactor = -gamma0 / (1 + alpha**2)

    mxH = np.cross(m_vec, H_vec)
    mxmxH = np.cross(m_vec, mxH)
    torque_llg = prefactor * (mxH + alpha * mxmxH)

    return torque_llg


def LLG_Heun(
    m_i: ntp.NDArray[np.float64],
    T: float,
    Vol: float,
    dt: float,
    alpha: float,
    gamma0: float = VACUUM_PERMEABILITY * GYROMAGNETIC_RATIO,
    stt_enable: bool = True,
    recompute_H_th: bool = True,
    recompute_H_eff: bool = True,
    **kwargs: Unpack[MaterialProps],
) -> ntp.NDArray[np.float64]:
    """Calculates the next time step magnetization using Heun's method, with the possibility to
    recompute the total field H_tot at the intermediate step.

    Parameters
    ----------
        T: float
            Temperature of the system (K)
        Vol: float
            Volume of the magnetic sample (m^3)
        dt: float
            Time step interval (s)
        alpha: float
            Damping coefficient (unit-less)
        gamma0: float
            VACUUM_PERMEABILITY * GYROMAGNETIC_RATIO
        stt_enable: bool
            enables STT effect
        recompute_H_th: bool
            compute a new value for the thermal field for the intermediate step
        recompute_H_eff: bool
            recompute the effective field for the magnetization at the intermediate point
        **kwargs: MaterialProps
            Dictionary defining the properties of the material.
    """

    # First evaluation
    if T > 0:
        H_th = compute_thermal_field(alpha, T, kwargs["M_s"], Vol, dt)
    else:
        H_th = 0
    H_eff = calc_Heff(m_i, stt_enable, **kwargs)
    H_tot = (
        H_eff + H_th
    )  # Note: the externally applied field H_app is included in H_eff
    k1 = LLG_rhs(m_i, H_tot, gamma0, alpha)
    m_temp = m_i + dt * k1
    m_temp /= np.linalg.norm(m_temp)  # Normalize after k1

    # Second evaluation
    if recompute_H_th and T > 0:
        # compute a new random thermal field for the intermediate step (which represents
        # a point in the successive time instant)
        H_th = compute_thermal_field(alpha, T, kwargs["M_s"], Vol, dt)
    if recompute_H_eff:
        # recomputing the effective field with the intermediate magnetization m_temp
        H_eff = calc_Heff(m_temp, True, **kwargs)
    if recompute_H_th or recompute_H_eff:
        H_tot = H_th + H_eff
    k2 = LLG_rhs(
        m_temp, H_tot, gamma0, alpha
    )  # Recalculate k2 with updated m_temp and H_tot

    # Heun step
    m_next = m_i + (dt / 2) * (k1 + k2)
    m_next /= np.linalg.norm(m_next)  # Renormalize

    return m_next

def LLG_Heun_Heff(
    m_i: ntp.NDArray[np.float64],
    T: float,
    Vol: float,
    dt: float,
    alpha: float,
    gamma0: float = VACUUM_PERMEABILITY * GYROMAGNETIC_RATIO,
    stt_enable: bool = True,
    recompute_H_th: bool = True,
    recompute_H_eff: bool = True,
    **kwargs: Unpack[MaterialProps],
) -> ntp.NDArray[np.float64]:
    """Calculates the next time step magnetization using Heun's method, with the possibility to
    recompute the total field H_tot at the intermediate step. Returns both m and the effective field for the previous step.

    Parameters
    ----------
        T: float
            Temperature of the system (K)
        Vol: float
            Volume of the magnetic sample (m^3)
        dt: float
            Time step interval (s)
        alpha: float
            Damping coefficient (unit-less)
        gamma0: float
            VACUUM_PERMEABILITY * GYROMAGNETIC_RATIO
        stt_enable: bool
            enables STT effect
        recompute_H_th: bool
            compute a new value for the thermal field for the intermediate step
        recompute_H_eff: bool
            recompute the effective field for the magnetization at the intermediate point
        **kwargs: MaterialProps
            Dictionary defining the properties of the material.
    """

    # First evaluation
    if T > 0:
        H_th = compute_thermal_field(alpha, T, kwargs["M_s"], Vol, dt)
    else:
        H_th = 0
    H_eff = calc_Heff(m_i, stt_enable, **kwargs)
    H_tot = (
        H_eff + H_th
    )  # Note: the externally applied field H_app is included in H_eff
    k1 = LLG_rhs(m_i, H_tot, gamma0, alpha)
    m_temp = m_i + dt * k1
    m_temp /= np.linalg.norm(m_temp)  # Normalize after k1

    # Second evaluation
    if recompute_H_th and T > 0:
        # compute a new random thermal field for the intermediate step (which represents
        # a point in the successive time instant)
        H_th = compute_thermal_field(alpha, T, kwargs["M_s"], Vol, dt)
    if recompute_H_eff:
        # recomputing the effective field with the intermediate magnetization m_temp
        H_eff = calc_Heff(m_temp, True, **kwargs)
    if recompute_H_th or recompute_H_eff:
        H_tot = H_th + H_eff
    k2 = LLG_rhs(
        m_temp, H_tot, gamma0, alpha
    )  # Recalculate k2 with updated m_temp and H_tot

    # Heun step
    m_next = m_i + (dt / 2) * (k1 + k2)
    m_next /= np.linalg.norm(m_next)  # Renormalize

    return m_next, H_eff
