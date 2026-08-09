"""Evaluate a trained neural surrogate (without Kalman filter)."""

import argparse
from wdn_kalman.datasets import DATASETS, get_dataset
from wdn_kalman.paths import ProjectPaths
from wdn_kalman.surrogate.evaluation import SurrogateEvaluator, save_surrogate_evaluation_result
from wdn_kalman.surrogate.manager import SurrogateManager


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate single step chlorine predictions of trained neural surrogate model."
    )

    parser.add_argument(
        "--dataset",
        choices=(*DATASETS.keys(), "all"),
        default="all",
        help="Dataset to evaluate.",
    )

    return parser.parse_args()


def main() -> None:
    """Evaluate and save neural surrogate model results."""
    args = parse_args()
    paths = ProjectPaths.from_project_root()
    paths.create_output_dirs()
    paths.validate_inputs()
    surrogate_manager = SurrogateManager(paths)
    evaluator = SurrogateEvaluator(surrogate_manager=surrogate_manager)

    if args.dataset == "all":
        datasets = list(DATASETS.values())
    else:
        datasets = [get_dataset(args.dataset)]

    for dataset in datasets:
        print(f"\nEvaluating surrogate for {dataset.net_name}")

        result = evaluator.run_neural_surrogate_evaluation(dataset)
        save_surrogate_evaluation_result(result)

        print(f"MAE (nodes + links): {result.mean_mae:.4f} ± {result.std_mae:.4f} mg/L")
        print(f"MAE (node only): {result.node_mean_mae:.4f} ± {result.node_std_mae:.4f} mg/L")
        print(f"MAE (link only): {result.link_mean_mae:.4f} ± {result.link_std_mae:.4f} mg/L")
        print(f"Saved surrogate evaluation to: {result.result_file}")


if __name__ == "__main__":
    main()
