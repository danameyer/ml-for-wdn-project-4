"""Merge GNN training sweep results."""

import argparse
import csv
import json
import numpy as np
from wdn_kalman.paths import ProjectPaths


BUFFER_SIZES = (1, 5, 10, 15, 20, 25)
ROLLOUTS = (1, 5, 10, 15, 20, 25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge GNN training sweep results."
    )

    parser.add_argument(
        "--dataset",
        choices=("net1", "hanoi"),
        default="net1",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_project_root()
    result_dir = paths.aggregated_results_dir / "gnn_training_sweep"
    result_dir.mkdir(parents=True, exist_ok=True)
    node_losses = np.full((len(ROLLOUTS), len(BUFFER_SIZES)), np.nan)
    total_losses = np.full_like(node_losses, np.nan)
    link_losses = np.full_like(node_losses, np.nan)
    spread_losses = np.full_like(node_losses, np.nan)
    missing_configs = []

    for buffer_index, buffer_size in enumerate(BUFFER_SIZES):
        for rollout_index, rollout in enumerate(ROLLOUTS):
            result_file = result_dir / f"{args.dataset}_buffer={buffer_size}_rollout={rollout}_seed={args.seed}.json"

            if not result_file.exists():
                missing_configs.append((buffer_size, rollout))
                continue

            with result_file.open("r", encoding="utf-8") as file:
                result = json.load(file)

            node_losses[rollout_index, buffer_index] = result["node_loss"]
            total_losses[rollout_index, buffer_index] = result["total_loss"]
            link_losses[rollout_index, buffer_index] = result["link_loss"]
            spread_losses[rollout_index, buffer_index] = result["spread_loss"]

    print("\nNode concentration reconstruction loss")
    print("Rows = rollout length, columns = buffer size\n")
    print("rollout", *[f"{buffer:>10}" for buffer in BUFFER_SIZES])

    for rollout_index, rollout in enumerate(ROLLOUTS):
        values = node_losses[rollout_index]

        print(f"{rollout:>7}", *[f"{value:10.0f}" for value in values])

    npz_file = result_dir / f"{args.dataset}_gnn_training_sweep.npz"

    np.savez(
        npz_file,
        buffer_sizes=np.asarray(BUFFER_SIZES),
        rollouts=np.asarray(ROLLOUTS),
        node_losses=node_losses,
        total_losses=total_losses,
        link_losses=link_losses,
        spread_losses=spread_losses,
    )

    csv_file = result_dir/ f"{args.dataset}_gnn_training_sweep_node_loss.csv"

    with csv_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["rollout", *BUFFER_SIZES])

        for rollout_index, rollout in enumerate(ROLLOUTS):
            writer.writerow([rollout, *node_losses[rollout_index].tolist()])

    print(f"\nSaved: {npz_file}")
    print(f"Saved: {csv_file}")

    if missing_configs:
        print("\nMissing configurations:")

        for buffer_size, rollout in missing_configs:
            print(f"buffer={buffer_size}, rollout={rollout}")


if __name__ == "__main__":
    main()