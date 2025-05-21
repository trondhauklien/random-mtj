from mtj.analysis import PerpSTT
import numpy as np


def main() -> None:
    analysis = PerpSTT(num_samples=200, volume=5e-9 ** 2 * 1e-9 * np.pi, volt_ampliude=0.75)
    analysis.do_simulations()
    analysis.plot_trajectories(range(5))
    analysis = PerpSTT(num_samples=200, volume=5e-9 ** 2 * 1e-9 * np.pi, volt_ampliude=0.75)
    analysis.m0 = -analysis.m0
    analysis.do_simulations()
    analysis.plot_trajectories(range(5))

    analysis = PerpSTT(num_samples=200, volume=50e-9 ** 2 * 1e-9 * np.pi, volt_ampliude=0.75)
    analysis.do_simulations()
    analysis.plot_trajectories(range(5))
    analysis = PerpSTT(num_samples=200, volume=50e-9 ** 2 * 1e-9 * np.pi, volt_ampliude=0.75)
    analysis.m0 = -analysis.m0
    analysis.do_simulations()
    analysis.plot_trajectories(range(5))


    analysis = PerpSTT(num_samples=200, volume=100e-9 ** 2 * 1e-9 * np.pi, volt_ampliude=0.75)
    analysis.do_simulations()
    analysis.plot_trajectories(range(5))
    analysis = PerpSTT(num_samples=200, volume=100e-9 ** 2 * 1e-9 * np.pi, volt_ampliude=0.75)
    analysis.m0 = -analysis.m0
    analysis.do_simulations()
    analysis.plot_trajectories(range(5))



if __name__ == "__main__":
    main()
