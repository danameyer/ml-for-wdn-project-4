"""Run an EnKF experiment."""

import argparse
from wdn_kalman.datasets import DATASETS, get_dataset
from wdn_kalman.enkf.experiment import EnKFExperiment
from wdn_kalman.paths import ProjectPaths
from wdn_kalman.results import save_experiment_result
from wdn_kalman.surrogate.manager import SurrogateManager
from wdn_kalman.enkf.gnn_experiment import GNNEnKFExperiment
from wdn_kalman.gnn.manager import GNNManager


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run an EnKF experiment.")

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
        help="State transition model used by the EnKF.",
    )

    parser.add_argument(
        "--n-iters",
        type=int,
        default=30,
        help="Number of random sensor placements per sensor count."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=50,
        help="Number of ensemble members.",
    )

    parser.add_argument(
        "--initial-variance",
        type=float,
        default=0.01,
        help="Initial state variance.",
    )

    parser.add_argument(
        "--process-variance",
        type=float,
        default=0.001,
        help="Process noise variance.",
    )

    parser.add_argument(
        "--measurement-variance",
        type=float,
        default=0.003,
        help="Measurement noise variance.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional maximum number of time steps. "
    )

    parser.add_argument(
        "--buffer-size",
        type=int,
        default=5,
        help="Buffer size of the GNN.",
    )

    parser.add_argument(
        "--unroll-steps",
        type=int,
        default=20,
        help="Unroll steps used when training the GNN.",
    )

    return parser.parse_args()


def main() -> None:
    """Run and save one EnKF experiment."""
    args = parse_args()

    if args.n_iters < 1:
        raise ValueError("--n-iters must be at least 1.")

    if args.ensemble_size < 2:
        raise ValueError("--ensemble-size must be at least 2.")

    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be at least 1.")

    dataset = get_dataset(args.dataset)
    paths = ProjectPaths.from_project_root()
    paths.create_output_dirs()

    if args.model == "surrogate":
        surrogate_manager = SurrogateManager(paths)
        experiment = EnKFExperiment(paths=paths, surrogate_manager=surrogate_manager)

        result = experiment.run(
            dataset=dataset,
            n_iters=args.n_iters,
            seed=args.seed,
            ensemble_size=args.ensemble_size,
            initial_variance=args.initial_variance,
            process_variance=args.process_variance,
            measurement_variance=args.measurement_variance,
            max_steps=args.max_steps,
        )

    else:
        gnn_manager = GNNManager(paths)
        experiment = GNNEnKFExperiment(paths=paths, gnn_manager=gnn_manager)

        result = experiment.run(
            dataset=dataset,
            n_iters=args.n_iters,
            seed=args.seed,
            buffer_size=args.buffer_size,
            unroll_steps=args.unroll_steps,
            ensemble_size=args.ensemble_size,
            initial_variance=args.initial_variance,
            process_variance=args.process_variance,
            measurement_variance=args.measurement_variance,
            max_steps=args.max_steps,
        )

    save_experiment_result(result=result, kalman_type="EnKF")
    print(f"Saved {args.model} EnKF result to: {result.result_file}")


if __name__ == "__main__":
    main()