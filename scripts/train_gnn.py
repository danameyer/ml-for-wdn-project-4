"""Train and save GNN."""

import argparse
from wdn_kalman.datasets import (DATASETS, get_dataset)
from wdn_kalman.gnn.manager import GNNManager
from wdn_kalman.paths import ProjectPaths


BUFFER_SIZE = 20
UNROLL_STEPS = 20
EPOCHS = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GNN.")
    parser.add_argument("--dataset", choices=(*DATASETS.keys(), "all"), default="all")
    parser.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
    parser.add_argument("--unroll-steps", type=int, default=UNROLL_STEPS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--link-loss-weight", type=float, default=1.0)
    parser.add_argument("--buffer-spread-weight", type=float,default=0.01)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_project_root()
    paths.create_output_dirs()
    manager = GNNManager(paths)

    if args.dataset == "all":
        datasets = list(DATASETS.values())
    else:
        datasets = [get_dataset(args.dataset)]

    for dataset in datasets:
        model_file = manager.train(
            dataset=dataset,
            buffer_size=args.buffer_size,
            unroll_steps=args.unroll_steps,
            epochs=args.epochs,
            lr=args.lr,
            link_loss_weight=args.link_loss_weight,
            buffer_spread_weight=args.buffer_spread_weight,
            overwrite=args.overwrite,
        )

        print(f"Saved GNN checkpoint to: {model_file}")


if __name__ == "__main__":
    main()
