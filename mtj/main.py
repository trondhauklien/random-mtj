from datetime import datetime
from pathlib import Path

import numpy as np

from mtj.calc_Heff import calc_Heff
from mtj.calc_Hth import compute_thermal_field
from mtj.init import init_m
from mtj.llg_heun import LLG_Heun


def main() -> None:
    # Get the initial magnetization
    Tn = 1e-11  # (s)
    dt = 1e-13  # time step (s)
    time_series = np.arange(0, Tn, dt)

    m = init_m(np.array([0, 1, 0]), len(time_series))

    base_path = Path(__file__).parent.parent
    output_dir = base_path / "output"
    output_dir.mkdir(exist_ok=True)
    output_file_path = (
        base_path / f"output/{datetime.now().strftime('%Y-%m-%d-%H%M%S')}-mag.csv"
    )

    alpha = 0.01  # damping factor
    T = 300  # Temperature (K)
    Ms = 800e3  # Saturation magnetization (A/m)
    V = 1e-27  # Volume of the magnetic particle (m^3)

    for i, t in enumerate(time_series[:-1]):
        H_th = compute_thermal_field(alpha, T, Ms, V, dt)
        print("Thermal Field Vector H_th:", H_th)
        # Calculate the effective field
        H_eff = calc_Heff(np.array([1, 0, 0]))

        # Calculate the magnetization for the next time step
        m[i + 1] = LLG_Heun(m[i], H_eff)
        pass

    np.savetxt(output_file_path, m, delimiter=",")


if __name__ == "__main__":
    main()
