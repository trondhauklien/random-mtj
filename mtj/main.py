from mtj.analysis import PerpSTT


def main() -> None:
    analysis = PerpSTT(random=True, num_samples=1000)

    analysis.do_simulations()
    analysis.plot_trajectories(range(5))


if __name__ == "__main__":
    main()
