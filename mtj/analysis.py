import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.progress import track

from mtj.calc_torque import calc_torque
from mtj.constants import VACUUM_PERMEABILITY, e, hbar
from mtj.llg_heun import LLG_Heun


class PerpSTT:
    Ms = 1000e3  # A/m
    p = np.array([0, 0, 1])
    m0 = np.array([0.11, 0.11, 0.9])  # Initial magnetization parallell to polarizer
    R_pp = 2e3  # Ohm
    volume = 50e-9 * 50e-9 * 1e-9  # m^3
    N = np.diag([0.029569, 0.029569, 0.940862])  # Corresponding to shape of free layer
    dt = 100e-15  # s
    u_k = np.array([0, 0, 1])  # Perpendicular MCA
    K_u = 10e5
    alpha = 0.01
    a_para = (
        hbar / (2 * e) * np.sqrt(3) / (4 * Ms * R_pp * volume * VACUUM_PERMEABILITY)
    )
    max_run_time = 8e-9
    time_series = np.arange(0, max_run_time, dt)
    time_steps = len(time_series)
    T = 300  # K
    H_app = np.array([0, 0, 0])

    def __init__(self, random=False, num_samples=10, volume=None):
        if random:
            self.rng = np.random.default_rng()
            self.voltages = self.rng.uniform(-1, 1, num_samples)
        else:
            self.voltages = np.linspace(-1, 1, num_samples)

        if volume:
            self.volume = volume

        self.m = np.zeros(shape=(num_samples, self.time_steps, 3), dtype=np.float32)
        self.torques = np.zeros(shape=(num_samples, self.time_steps), dtype=np.float32)

        self.start_date = f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"

        # self.progress = Progress(
        #     "[progress.description]{task.description}",
        #     BarColumn(),
        #     "[progress.percentage]{task.percentage:>3.0f}%",
        #     TimeRemainingColumn(),
        #     TimeElapsedColumn(),
        #     refresh_per_second=1,  # bit slower updates
        # )
        # self.progress_task = self.progress.add_task(
        #     "[green]Progress:", total=num_samples
        # )

    def do_simulations(self):
        simulation_start = time.time()
        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    self.calculate_magnetization, self.m, self.voltages, self.torques
                )
            )

        # Unpack results into self.m and self.torque
        self.m, self.torques = zip(*results)
        self.m = np.array(self.m)
        self.torques = np.array(self.torques)

        simulation_end = time.time()

        simulation_time = simulation_end - simulation_start

        print(f"Simulation took {simulation_time:.2f} s")

        output_dir = self.get_ouput_dir()
        np.save(output_dir / "mag", self.m)
        np.save(output_dir / "voltages", self.voltages)
        np.save(output_dir / "time_series", self.time_series)
        np.save(output_dir / "torques", self.torques)

    def calculate_magnetization(self, m, v, torque):
        m0 = self.m0 if v < 0 else -self.m0
        m[0] = m0 / np.linalg.norm(m0)

        for i, _ in enumerate(self.time_series[:-1]):
            # Calculate the magnetization for the next time step
            m[i + 1], H_eff = LLG_Heun(
                m_i=m[i],
                T=self.T,
                Vol=self.volume,
                dt=self.dt,
                alpha=self.alpha,
                H_app=self.H_app,
                M_s=self.Ms,
                K_u=self.K_u,
                u_k=self.u_k,
                N=self.N,
                p=self.p,
                a_para=self.a_para,
                a_ortho=0,
                V=v,
            )
            torque[i + 1] = calc_torque(m[i + 1], H_eff, self.Ms)

        return m, torque

    def get_ouput_dir(self):
        base_path = Path(__file__).parent.parent
        output_dir = base_path / "output" / f"PerpSTT-{self.start_date}"
        output_dir.mkdir(exist_ok=True)

        return output_dir

    def plot_trajectories(self, indices=[0]):
        fig = plt.figure(figsize=(15, 7))
        for i in indices:
            sphere_ax = fig.add_subplot(2, len(indices), i + 1, projection="3d")
            PerpSTT.plot_unit_sphere(sphere_ax, self.m[i], "m", f"{self.voltages[i]} V")

            ax = fig.add_subplot(2, len(indices), len(indices) + 1 + i)
            ax.plot(self.time_series, self.torques[i])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("|m x Heff|/ Ms")

        output_dir = self.get_ouput_dir()
        fig.savefig(output_dir / "trajectories.png")

    @staticmethod
    def plot_unit_sphere(ax, m, label, title):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        ax.quiver(0, 0, 0, m[0, 0], m[0, 1], m[0, 2], color="red", lw=1, label="$m_0$")
        ax.plot_surface(x, y, z, color="lightgray", alpha=0.2, rstride=10, cstride=10)
        ax.plot(m[:, 0], m[:, 1], m[:, 2], label=label, lw=1)

        ax.set_xlabel(r"$m_x$")
        ax.set_ylabel(r"$m_y$")
        ax.set_zlabel(r"$m_z$")
        ax.set_title(f"{title}")
        ax.set_box_aspect([1, 1, 1])
        if label:
            ax.legend()
