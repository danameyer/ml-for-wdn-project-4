"""Train neural surrogate model."""

import argparse

from wdn_kalman.datasets import (
    DATASETS,
    get_dataset,
)
from wdn_kalman.paths import ProjectPaths
from wdn_kalman.surrogate.manager import (
    SurrogateManager,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train neural surrogate model."
    )

    parser.add_argument(
        "--dataset",
        choices=(*DATASETS.keys(), "all"),
        default="all",
        help="Dataset to train.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain even when model files already exist.",
    )

    return parser.parse_args()


def main() -> None:
    """Train the requested surrogate model or models."""
    args = parse_args()

    paths = ProjectPaths.from_project_root()
    paths.create_output_dirs()

    surrogate_manager = SurrogateManager(paths)

    if args.dataset == "all":
        datasets = list(DATASETS.values())
    else:
        datasets = [get_dataset(args.dataset)]

    for dataset in datasets:
        print(f"\nTraining surrogate for {dataset.net_name}")

        surrogate_manager.train(
            dataset=dataset,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()