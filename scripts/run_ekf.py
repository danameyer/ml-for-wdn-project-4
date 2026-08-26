"""Run an EKF experiment."""

import argparse
from wdn_kalman.datasets import DATASETS, get_dataset
from wdn_kalman.ekf.experiment import EKFExperiment
from wdn_kalman.paths import ProjectPaths
from wdn_kalman.results import save_experiment_result
from wdn_kalman.surrogate.manager import SurrogateManager
from wdn_kalman.ekf.gnn_experiment import GNNEKFExperiment
from wdn_kalman.gnn.manager import GNNManager


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
        "--model",
        choices=("surrogate", "gnn"),
        default="surrogate",
        help="State transition model used.",
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

    parser.add_argument(
        "--buffer-size",
        type=int,
        default=5,
        help="Buffer size of the trained GNN.",
    )

    parser.add_argument(
        "--unroll-steps",
        type=int,
        default=20,
        help="Unroll steps used when training the GNN.",
    )

    parser.add_argument(
        "--initial-variance",
        type=float,
        default=0.01,
        help="Initial EKF state variance.",
    )

    parser.add_argument(
        "--process-variance",
        type=float,
        default=0.001,
        help="EKF process noise variance.",
    )

    parser.add_argument(
        "--measurement-variance",
        type=float,
        default=0.01,
        help="EKF measurement noise variance.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of time steps.",
    )

    return parser.parse_args()


def main() -> None:
    """Run and save an EKF experiment."""
    args = parse_args()

    dataset = get_dataset(args.dataset)

    paths = ProjectPaths.from_project_root()
    paths.create_output_dirs()

    if args.model == "surrogate":
        surrogate_manager = SurrogateManager(paths)
        experiment = EKFExperiment(paths=paths, surrogate_manager=surrogate_manager)
        result = experiment.run(dataset=dataset, n_iters=args.n_iters, seed=args.seed)

    else:
        gnn_manager = GNNManager(paths)
        experiment = GNNEKFExperiment(paths=paths, gnn_manager=gnn_manager)

        result = experiment.run(
            dataset=dataset,
            n_iters=args.n_iters,
            seed=args.seed,
            buffer_size=args.buffer_size,
            unroll_steps=args.unroll_steps,
            initial_variance=args.initial_variance,
            process_variance=args.process_variance,
            measurement_variance=args.measurement_variance,
            max_steps=args.max_steps,
        )

    save_experiment_result(result=result, kalman_type="EKF",)
    print(f"Saved {args.model} EKF result to: {result.result_file}")


if __name__ == "__main__":
    main()