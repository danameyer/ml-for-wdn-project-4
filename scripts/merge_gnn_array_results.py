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
        default=50
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_project_root()
    dataset = get_dataset(args.dataset)
    filter_type = args.filter_type.lower()
    seeds = np.arange(args.start_seed, args.start_seed + args.n_iters)
    placement_scores = []
    placement_stds = []
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
            placement_stds.append(result["std_scores"].copy())

    placement_scores = np.vstack(placement_scores)
    placement_stds = np.vstack(placement_stds)

    finite_mask = np.isfinite(placement_scores) & np.isfinite(placement_stds)
    valid_placement_counts = np.sum(finite_mask, axis=0)
    failed_placement_counts = args.n_iters - valid_placement_counts

    mean_scores = np.full(len(sensors), np.nan, dtype=float)

    std_scores = np.full(len(sensors), np.nan, dtype=float)

    for sensor_index in range(len(sensors)):
        valid_mask = finite_mask[:, sensor_index]
        valid_means = placement_scores[valid_mask, sensor_index]
        valid_stds = placement_stds[valid_mask, sensor_index]

        if len(valid_means) == 0:
            continue

        pooled_mean = np.mean(valid_means)
        pooled_second_moment = np.mean(valid_stds**2 + valid_means**2)
        pooled_variance = pooled_second_moment - pooled_mean**2
        mean_scores[sensor_index] = pooled_mean
        std_scores[sensor_index] = np.sqrt(max(pooled_variance, 0.0))

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
        "placement_mean_scores": placement_scores.T,
        "placement_std_scores": placement_stds.T,
        "placement_seeds": seeds,
        "valid_placement_counts": valid_placement_counts,
        "failed_placement_counts": failed_placement_counts,
    }

    if filter_type == "enkf":
        saved_values["ensemble_size"] = args.ensemble_size

    np.savez_compressed(output_file, **saved_values)

    print("Merged result:")
    print(output_file)
    print()

    for sensor_count, mean, std, valid_count in zip(sensors, mean_scores, std_scores, valid_placement_counts):
        print(f"{sensor_count} sensors/type: {mean:.4f} ± {std:.4f} ({valid_count}/{args.n_iters} valid)")

    print()
    print("Largest placement scores:")

    for sensor_column, sensor_count in enumerate(sensors):
        values = placement_scores[:, sensor_column]
        valid_indices = np.where(np.isfinite(values))[0]

        if len(valid_indices) == 0:
            print(f"{sensor_count} sensors/type: no finite placements")
            continue

        worst_local_index = int(np.argmax(values[valid_indices]))
        worst_index = int(valid_indices[worst_local_index])
        print(f"{sensor_count} sensors/type: max={values[worst_index]:.4f}, seed={seeds[worst_index]}")


if __name__ == "__main__":
    main()
