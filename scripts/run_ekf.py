"""Run an Extended Kalman filter experiment."""

import argparse

from wdn_kalman.datasets import (
    DATASETS,
    get_dataset,
)
from wdn_kalman.ekf.experiment import (
    EKFExperiment,
)
from wdn_kalman.paths import ProjectPaths
from wdn_kalman.results import (
    save_experiment_result,
)
from wdn_kalman.surrogate.manager import (
    SurrogateManager,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run an EKF experiment."
    )

    parser.add_argument(
        "--dataset",
        choices=tuple(DATASETS),
        required=True,
        help="Dataset used for the experiment.",
    )

    parser.add_argument(
        "--n-iters",
        type=int,
        default=30,
        help=(
            "Number of random sensor placements "
            "per sensor count."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def main() -> None:
    """Run and save an EKF experiment."""
    args = parse_args()

    dataset = get_dataset(args.dataset)

    paths = ProjectPaths.from_project_root()
    paths.create_output_dirs()

    surrogate_manager = SurrogateManager(
        paths
    )

    experiment = EKFExperiment(
        paths=paths,
        surrogate_manager=surrogate_manager,
    )

    result = experiment.run(
        dataset=dataset,
        n_iters=args.n_iters,
        seed=args.seed,
    )

    save_experiment_result(
        result=result,
        kalman_type="EKF",
    )

    print(
        f"Saved EKF result to:\n"
        f"{result.result_file}"
    )


if __name__ == "__main__":
    main()