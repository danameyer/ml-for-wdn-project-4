"""Plot a Kalman-filter result."""

import argparse
from pathlib import Path

from wdn_kalman.datasets import HANOI, NET1
from wdn_kalman.paths import ProjectPaths
from wdn_kalman.plotting import BaselinePlot
from wdn_kalman.results import (
    load_experiment_result,
)


DATASETS = {
    "net1": NET1,
    "hanoi": HANOI,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot a Kalman-filter result."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=DATASETS.keys(),
        required=True,
    )

    parser.add_argument(
        "--kalman-type",
        choices=("EKF", "EnKF"),
        required=True,
    )

    parser.add_argument(
        "--n-iters",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--result-file",
        type=Path,
        default=None,
        help=(
            "Optional explicit result file. "
            "Otherwise, the path is constructed "
            "from the other arguments."
        ),
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help=(
            "Save the plot without opening a "
            "plot window."
        ),
    )

    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=50,
        help=(
            "Ensemble size used for an EnKF result. "
            "Default: 50."
        ),
    )

    parser.add_argument(
        "--model",
        choices=("surrogate", "gnn"),
        default="surrogate",
    )

    return parser.parse_args()


def main() -> None:
    """Load and plot an experiment result."""
    args = parse_args()

    paths = ProjectPaths.from_project_root()
    paths.create_output_dirs()

    dataset = DATASETS[args.dataset]

    if args.result_file is None:
        if args.model == "gnn":
            model_name = "gnn_"
        else:
            model_name = ""

        if args.kalman_type == "EnKF":
            result_filename = (
                f"{dataset.file_prefix}_"
                f"{model_name}"
                f"enkf_"
                f"n_iters={args.n_iters}_"
                f"ensemble_size="
                f"{args.ensemble_size}_"
                f"seed={args.seed}.npz"
            )
        else:
            result_filename = (
                f"{dataset.file_prefix}_"
                f"{model_name}"
                f"ekf_"
                f"n_iters={args.n_iters}_"
                f"seed={args.seed}.npz"
            )

        result_file = paths.aggregated_results_dir / result_filename
    else:
        result_file = args.result_file

    result = load_experiment_result(
        input_file=result_file,
        dataset=dataset,
    )

    plotter = BaselinePlot(paths)

    plotter.plot_experiment_results(
        result=result,
        kalman_type=args.kalman_type,
        model=args.model,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()