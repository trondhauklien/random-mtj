import streamlit as st
import numpy as np

from mtj.types import MaterialProps
from mtj.constants import VACUUM_PERMEABILITY

import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import time

from mtj.init import init_m
from mtj.llg_heun import LLG_Heun

st.title("Introduction / Sandbox")
orientation_axes = {
    "x direction (1, 0, 0)": np.array([1, 0, 0]),
    "y direction (0, 1, 0)": np.array([0, 1, 0]),
    "z direction (0, 0, 1)": np.array([0, 0, 1]),
    "xy plane (1, 1, 0)": np.array([1, 1, 0]),
}

demag_tensor = {"thin film diag(0, 0, 1)": np.diag([0, 0, 1])}

params: MaterialProps = {
    "K_u": 5e5,
    "M_s": 1 / VACUUM_PERMEABILITY,
    "u_k": orientation_axes["z direction (0, 0, 1)"],
    "p": orientation_axes["z direction (0, 0, 1)"],
    "a_para": 0,
    "a_ortho": 0,
    "V": 0,
    "H_app": 0,
    "N": demag_tensor["thin film diag(0, 0, 1)"],
}

Tn = 5e-9  # (s)
dt = 1e-12  # time step (s)
plotting_speed = 50

m0 = np.array([0.1, 0.1, 1], dtype=np.float32)


def normalize_m0(m0: str):
    interpreted = np.fromstring(m0, dtype=np.float32, sep=" ")
    noisy = interpreted + np.random.normal(0, 0.05, 3)
    return noisy / np.linalg.norm(noisy)


with st.sidebar:
    m0 = normalize_m0(st.text_input("$m_0$", "0 0 1"))

    with st.form("addition"):
        with st.expander("Mag. Parameters"):
            params["N"] = demag_tensor[
                st.selectbox("$N$", options=demag_tensor.keys(), index=0)
            ]
            params["M_s"] = st.number_input(
                "$M_s$", value=params["M_s"], format="%0.1e"
            )
            params["K_u"] = st.number_input(
                "$K_u$", value=params["K_u"], format="%0.1e"
            )
            params["u_k"] = orientation_axes[
                st.selectbox("$u_k$", options=orientation_axes.keys(), index=2)
            ]
            params["p"] = orientation_axes[
                st.selectbox("$p$", options=list(orientation_axes.keys())[:-1], index=2)
            ]
            H_app = st.number_input("$H_{app}$", value=0)
            h_app = orientation_axes[
                st.selectbox(
                    "$h_{app}$",
                    options=list(orientation_axes.keys())[:-1],
                    index=2,
                )
            ]
        Tn = st.number_input("End Time", min_value=0.0, value=Tn, format="%0.1e")
        dt = st.number_input(
            r"Step Size ($\Delta t$)", min_value=0.0, value=dt, format="%0.1e"
        )
        plotting_speed = st.slider(
            "Plotting Speed", min_value=1, max_value=100, step=1, value=plotting_speed
        )

        submit = st.form_submit_button("add")


def plot_unit_sphere(ax, m, label):
    u = np.linspace(0, 2 * np.pi, 200)
    v = np.linspace(0, np.pi, 200)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    arrow = ax.quiver(
        0, 0, 0, m[-1, 0], m[-1, 1], m[-1, 2], color="red", lw=1, label="$m$"
    )
    ax.plot_surface(x, y, z, color="lightgray", alpha=0.2, rstride=10, cstride=10)
    (line,) = ax.plot(m[:, 0], m[:, 1], m[:, 2], label=label, lw=2)

    ax.set_xlabel(r"$m_x$")
    ax.set_ylabel(r"$m_y$")
    ax.set_zlabel(r"$m_z$")
    ax.set_title(f"Magnetization {label}")
    ax.set_box_aspect([1, 1, 1])
    ax.legend()

    return line, arrow


alpha = 0.008  # Damping factor (arbitrarily chosen in this demo)
T = 0  # Temperature (K) - H_th diabled if 0
Vol = 1e-9 * 25e-9**2 * np.pi  # Volume
stt_enable = False
recompute_H_th = False
recompute_H_eff = False
time_series = np.arange(0, Tn, dt)
m = init_m(m0, int(Tn / dt))

fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
line, arrow = plot_unit_sphere(ax, m, "Trajectory")

plot_placeholder = st.empty()
text_placeholder = st.empty()

with plot_placeholder:
    st.pyplot(fig)


if submit:
    for i, t in enumerate(time_series[:-1]):
        # plt.close()
        m[i + 1] = LLG_Heun(
            m[i],
            T,
            Vol,
            dt,
            alpha,
            stt_enable=stt_enable,
            recompute_H_th=recompute_H_th,
            recompute_H_eff=recompute_H_eff,
            **params,
        )
        line.set_data_3d(m[: i + 1, :].T)
        new_mag = np.array([[0, 0, 0], m[i + 1, :]])
        arrow.set_segments([new_mag])

        if i % plotting_speed == 0:
            with plot_placeholder:
                st.pyplot(fig)
            with text_placeholder:
                st.progress(
                    i / len(time_series),
                    f"Simulating {i / len(time_series) * 100:.2f}%",
                )
