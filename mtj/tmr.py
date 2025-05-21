import numpy as np


def magnetization_to_resistance(m_array, R_P=2000, R_AP=500, p=np.array([0, 0, 1])):
    G_0 = 1 / R_P
    G_180 = 1 / R_AP
    ΔG = G_0 - G_180
    G_90 = (G_0 + G_180) / 2
    dot_product = m_array @ p
    G_series = G_90 + 0.5 * ΔG * dot_product
    R_series = 1 / G_series
    return R_series
