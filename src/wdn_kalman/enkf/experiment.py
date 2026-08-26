"""Ensemble Kalman filter experiment."""

import random

import numpy as np

from wdn_kalman.datasets import Dataset
from wdn_kalman.enkf.estimator import (
    EnKFExperimentConfig,
    EnKFStateEstimator,
)
from wdn_kalman.paths import ProjectPaths
from wdn_kalman.results import ExperimentResult
from wdn_kalman.surrogate.manager import SurrogateManager


class EnKFExperiment:
    """Run EnKF state-estimation experiment."""

    def __init__(
        self,
        paths: ProjectPaths,
        surrogate_manager: SurrogateManager,
    ):
        self.paths = paths
        self.surrogate_manager = surrogate_manager

    def run(
        self,
        dataset: Dataset,
        n_iters: int,
        seed: int = 0,
        ensemble_size: int = 50,
        initial_variance: float = 0.01,
        process_variance: float = 0.001,
        measurement_variance: float = 0.003,
        max_steps: int | None = None,
    ) -> ExperimentResult:
        """Run EnKF for all sensor counts and placements."""
        self.surrogate_manager.validate(dataset)
        random.seed(seed)
        np.random.seed(seed)

        mean_scores = []
        std_scores = []

        print(
            f"Running {dataset.net_name} EnKF: "
            f"{len(dataset.sensors)} sensor counts × "
            f"{n_iters} sensor placements"
        )

        for n_sensors in dataset.sensors:
            placement_scores = []

            for _ in range(n_iters):
                config = EnKFExperimentConfig(
                    num_node_quality_sensors=n_sensors,
                    num_link_sensors=n_sensors,
                    ensemble_size=ensemble_size,
                    initial_variance=initial_variance,
                    process_variance=process_variance,
                    measurement_variance=measurement_variance,
                    seed=seed,
                )

                estimator = EnKFStateEstimator(
                    surrogate_manager=self.surrogate_manager,
                    dataset=dataset,
                    config=config,
                )

                result = estimator.run(
                    max_steps=max_steps,
                )

                placement_scores.append(
                    np.asarray(result.chlorine_scores)
                )

            combined_scores = np.concatenate(
                [
                    scores.reshape(-1)
                    for scores in placement_scores
                ]
            )

            mean_score = float(
                np.mean(combined_scores)
            )

            std_score = float(
                np.std(combined_scores)
            )

            mean_scores.append(mean_score)
            std_scores.append(std_score)

            print(
                f"{n_sensors} sensors/type: "
                f"{mean_score:.4f} ± "
                f"{std_score:.4f}"
            )

        mean_scores = np.asarray(mean_scores)
        std_scores = np.asarray(std_scores)

        output_file = (
                self.paths.aggregated_results_dir
                / (
                    f"{dataset.file_prefix}_enkf_"
                    f"n_iters={n_iters}_"
                    f"ensemble_size={ensemble_size}_"
                    f"seed={seed}.npz"
                )
        )

        return ExperimentResult(
            dataset=dataset,
            sensors=dataset.sensors,
            mean_scores=mean_scores,
            std_scores=std_scores,
            n_iters=n_iters,
            seed=seed,
            result_file=output_file,
            ensemble_size=ensemble_size,
        )