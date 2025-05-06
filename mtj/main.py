from datetime import datetime
from pathlib import Path

import numpy as np

from mtj.init import init_m
from mtj.llg_heun import LLG_Heun


def main() -> None:
    # Get the initial magnetization
    Tn = 1e-11  # (s)
    dt = 1e-13  # time step (s)
    time_series = np.arange(0, Tn, dt)

    m = init_m(np.array([0, 0, 1]), len(time_series))
    p = np.array([0, 0, 1], dtype=np.float64)

    base_path = Path(__file__).parent.parent
    output_dir = base_path / "output"
    output_dir.mkdir(exist_ok=True)
    output_file_path = (
        base_path / f"output/{datetime.now().strftime('%Y-%m-%d-%H%M%S')}-mag.csv"
    )

    alpha = 0.01  # damping factor (unit-less)
    T = 300  # Temperature (K)
    M_s = 800e3  # Saturation magnetization module (A/m)
    Vol = 1e-27  # Volume of the magnetic particle (m^3)
    a_par = 1e-3  # A_parallel(V), in (T)
    a_perp = 2e-4  # A_perpendicular(V), in (T)
    K_u = 0 # crystal anisotropy constant
    u_k = np.array([0, 0, 1]) # easy axis
    Volt = 0 # STT voltage applied (V)
    H_app = np.array([0, 0, 0]) # applied field (A/m)
    N = np.array([[1,0,0],[0,1,0],[0,0,1]]) # demagnetization tensor

    for i, t in enumerate(time_series[:-1]):
        params = {
           "K_u": K_u,
            "M_s": M_s,
            "u_k": u_k,
            "p": p,
            "a_para": a_par,
            "a_ortho": a_perp,
            "V": Volt,
            "H_app": H_app,
            "N": N}

        # Calculate the magnetization for the next time step
        m[i + 1] = LLG_Heun(
            m[i],
            params,
            T,
            Vol,
            dt,
            alpha,
            recompute_H_th=True,
            recompute_H_eff=True
        )

    np.savetxt(output_file_path, m, delimiter=",")


if __name__ == "__main__":
    main()
