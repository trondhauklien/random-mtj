import numpy as np
import matplotlib.pyplot as plt
gamma = 1.76e11 #GyromagneticRatio (rad/s/T)
alpha = 0.01 #DampingCoefficient
k_B = 1.38e-23 #BoltzmannConstant (J/K)
T = 300 #Temperature (K)
M_s = 1.0e6 #SaturationMagnetization (A/m)
V = 1.0e-24 #VolumeoftheMagneticElement (m^3)
mu_0 = 4 * np.pi * 1e-7 #PermeabilityofFreeSpace (H/m)
H_eff = np.array([0, 0, 0.1]) #EffectiveField (Tesla)
dt = 1e-12 #Time step (s)
num_steps = 10000 #NumberofIterations
#Initial magnetization (normalized)
M = np.array([1, 0, 0])
M = M / np.linalg.norm(M)
sigma = np.sqrt(2 * alpha * k_B T / (gamma * M_s * V)) #ThermalFieldVariance
M_history = np.zeros((num_steps, 3)) #StoringResults
def llg_rhs(M, H_total):
    """Computes the RHS of the LLG equation using Runge-Kutta."""
    torque1 = -gamma * np.cross(M, H_total)
    torque2 = alpha * np.cross(M, torque1)
    return torque1 + torque2
#Stochastic LLG Solver using 4th-order Runge-Kutta
for i in range(num_steps):
    H_th = sigma * np.random.randn(3) / np.sqrt(dt) #Gaussiannoiseforthermalfield
    H_total = H_eff + H_th

    k1 = llg_rhs(M, H_total) * dt
    k2 = llg_rhs(M + k1/2, H_total) * dt
    k3 = llg_rhs(M + k2/2, H_total) * dt
    k4 = llg_rhs(M + k3, H_total) * dt

    M = M + (k1 + 2*k2 + 2*k3 + k4) / 6
    M = M / np.linalg.norm(M) #Normalizetomaintain |M| = 1

    M_history[i, :] = M #StoringResults

#PlottingResults
plt.figure(figsize=(8, 5))
plt.plot(np.arange(num_steps) * dt * 1e9, M_history[:, 0], label='M_x')
plt.plot(np.arange(num_steps) * dt * 1e9, M_history[:, 1], label='M_y')
plt.plot(np.arange(num_steps) * dt * 1e9, M_history[:, 2], label='M_z')
plt.xlabel('Time (ns)')
plt.ylabel('Magnetization Components')
plt.legend()
plt.title('Stochastic LLG Magnetization Dynamics (Runge-Kutta)')
plt.show()
