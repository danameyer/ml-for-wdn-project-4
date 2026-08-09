"""Plot a saved neural surrogate evaluation result."""

import argparse
from pathlib import Path
from wdn_kalman.datasets import DATASETS, get_dataset
from wdn_kalman.paths import ProjectPaths
from wdn_kalman.plotting import SurrogatePlot
from wdn_kalman.surrogate.evaluation import (
    load_surrogate_evaluation_result,
    surrogate_evaluation_file,
)

ANALYZED_NODES = {"net1": "12", "hanoi": "32"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare true and predicted chlorine for one node.")
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument(
        "--node-id",
        type=str,
        default=None,
        help="Node to plot. Default values: Net1: 12, Hanoi: 32.",
    )

    parser.add_argument(
        "--result-file",
        type=Path,
        default=None,
        help="Optional surrogate evaluation result file.",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the plot without opening window.",
    )

    return parser.parse_args()


def main() -> None:
    """Load and plot a surrogate evaluation result."""
    args = parse_args()
    dataset = get_dataset(args.dataset)
    paths = ProjectPaths.from_project_root()
    paths.create_output_dirs()

    if args.result_file is None:
        result_file = surrogate_evaluation_file(paths=paths, dataset=dataset)
    else:
        result_file = args.result_file

    evaluation_result = load_surrogate_evaluation_result(input_file=result_file, dataset=dataset)

    node_id = (
        args.node_id
        if args.node_id is not None
        else ANALYZED_NODES[args.dataset]
    )

    plotter = SurrogatePlot(paths)
    plotter.plot_node_prediction(result=evaluation_result,
                                 node_id=node_id,
                                 show=not args.no_show,
                                 max_steps=200)


if __name__ == "__main__":
    main()
