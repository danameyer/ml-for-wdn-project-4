"""Evaluate a trained GNN without a Kalman filter."""

import argparse

from wdn_kalman.datasets import (
    DATASETS,
    get_dataset,
)
from wdn_kalman.gnn.evaluation import (
    GNNEvaluator,
    save_gnn_evaluation_result,
)
from wdn_kalman.gnn.manager import (
    GNNManager,
)
from wdn_kalman.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate standalone GNN "
            "chlorine predictions."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=(
            *DATASETS.keys(),
            "all",
        ),
        default="all",
    )

    parser.add_argument(
        "--buffer-size",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--unroll-steps",
        type=int,
        default=20,
    )

    return parser.parse_args()


def main() -> None:
    """Evaluate and save GNN results."""
    args = parse_args()

    paths = (
        ProjectPaths.from_project_root()
    )

    paths.create_output_dirs()

    manager = GNNManager(
        paths
    )

    evaluator = GNNEvaluator(
        gnn_manager=manager
    )

    if args.dataset == "all":
        datasets = list(
            DATASETS.values()
        )
    else:
        datasets = [
            get_dataset(
                args.dataset
            )
        ]

    for dataset in datasets:
        print(
            f"\nEvaluating GNN for "
            f"{dataset.net_name}"
        )

        result = evaluator.run(
            dataset=dataset,
            buffer_size=(
                args.buffer_size
            ),
            unroll_steps=(
                args.unroll_steps
            ),
        )

        save_gnn_evaluation_result(
            result
        )

        print(
            "MAE (nodes + links): "
            f"{result.mean_mae:.4f} ± "
            f"{result.std_mae:.4f} mg/L"
        )

        print(
            "MAE (node only): "
            f"{result.node_mean_mae:.4f} ± "
            f"{result.node_std_mae:.4f} mg/L"
        )

        print(
            "MAE (link only): "
            f"{result.link_mean_mae:.4f} ± "
            f"{result.link_std_mae:.4f} mg/L"
        )

        print(
            "Saved GNN evaluation to: "
            f"{result.result_file}"
        )


if __name__ == "__main__":
    main()