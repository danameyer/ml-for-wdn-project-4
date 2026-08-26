"""Merge independent GNN Kalman sensor placement experiment results."""

import argparse
import numpy as np
from wdn_kalman.datasets import get_dataset
from wdn_kalman.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--filter-type",
        choices=("ekf", "enkf"),
        required=True
    )

    parser.add_argument(
        "--dataset",
        choices=("net1", "hanoi"),
        required=True
    )

    parser.add_argument(
        "--start-seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--n-iters",
        type=int,
        default=30
    )

    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=5
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_project_root()
    dataset = get_dataset(args.dataset)
    filter_type = args.filter_type.lower()
    seeds = np.arange(args.start_seed, args.start_seed + args.n_iters)
    placement_scores = []
    sensors = None
    dataset_name = None

    for seed in seeds:
        if filter_type == "ekf":
            result_filename = f"{dataset.file_prefix}_gnn_ekf_n_iters=1_seed={seed}.npz"
        else:
            result_filename = f"{dataset.file_prefix}_gnn_enkf_n_iters=1_ensemble_size={args.ensemble_size}_seed={seed}.npz"

        result_file = paths.aggregated_results_dir / result_filename

        if not result_file.exists():
            raise FileNotFoundError(result_file)

        with np.load(result_file, allow_pickle=False) as result:

            if sensors is None:
                sensors = result["sensors"].copy()

                dataset_name = str(result["dataset"].item())

            placement_scores.append(result["mean_scores"].copy())

    placement_scores = np.vstack(placement_scores)
    mean_scores = np.mean(placement_scores, axis=0)
    std_scores = np.std(placement_scores, axis=0)

    if filter_type == "ekf":
        output_filename = f"{dataset.file_prefix}_gnn_ekf_n_iters={args.n_iters}_seed={args.start_seed}.npz"
    else:
        output_filename = f"{dataset.file_prefix}_gnn_enkf_n_iters={args.n_iters}_ensemble_size={args.ensemble_size}_seed={args.start_seed}.npz"

    output_file = paths.aggregated_results_dir / output_filename
    saved_values = {
        "filter_name": filter_type.upper(),
        "dataset": dataset_name,
        "sensors": sensors,
        "mean_scores": mean_scores,
        "std_scores": std_scores,
        "n_iters": args.n_iters,
        "seed": args.start_seed,
        "placement_scores": placement_scores,
        "placement_seeds": seeds,
    }

    if filter_type == "enkf":
        saved_values["ensemble_size"] = args.ensemble_size

    np.savez_compressed(output_file, **saved_values)

    print("Merged result:")
    print(output_file)
    print()

    for sensor_count, mean, std in zip(sensors, mean_scores, std_scores):
        print(f"{sensor_count} sensors/type: {mean:.4f} ± {std:.4f}")

    print()
    print("Largest placement scores:")

    for sensor_column, sensor_count in enumerate(sensors):
        values = placement_scores[:, sensor_column]
        worst_index = int(np.argmax(values))

        print(f"{sensor_count} sensors/type: max={values[worst_index]:.4f}, seed={seeds[worst_index]}")


if __name__ == "__main__":
    main()