from datetime import datetime
from pathlib import Path

import numpy as np

from mtj.calc_Heff import calc_Heff
from mtj.init import init_m
from mtj.llg_heun import LLG_Heun


def main() -> None:
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

    for i, t in enumerate(T[:-1]):
        # Calculate the effective field
        H_eff = calc_Heff("1, 0, 0")

        # Calculate the magnetization for the next time step
        m[i + 1] = LLG_Heun(m[i], H_eff)
        pass

    np.savetxt(output_file_path, m, delimiter=",")


if __name__ == "__main__":
    main()
