import numpy as np
from calc_Heff import calc_Heff
from init import init_m
from llg_heun import LLG_Heun
from pathlib import Path
from datetime import datetime


def main():
    # Get the initial magnetization
    dt = 1e-9  # seconds
    Tn = 1e-6  # seconds
    T = np.arange(0, Tn, dt)  # time series

    m = init_m(np.array([0, 1, 0]), len(T))

    base_path = Path(__file__).parent.parent
    output_dir = base_path / "output"
    output_dir.mkdir(exist_ok=True)
    output_file_path = (
        base_path / f"output/{datetime.now().strftime('%Y-%m-%d-%H%M%S')}-mag.csv"
    )

    for i, t in enumerate(T):
        # Calculate the effective field
        H_eff = calc_Heff(m)

        # Calculate the magnetization for the next time step
        m[i + 1] = LLG_Heun(m[i], H_eff)
        pass

    np.savetxt(output_file_path, m, delimiter=",")


if __name__ == "__main__":
    main()
