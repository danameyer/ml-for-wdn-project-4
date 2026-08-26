"""GNN-based EnKF experiment."""

import random
import numpy as np
from wdn_kalman.datasets import Dataset
from wdn_kalman.enkf.gnn_estimator import GNNEnKFConfig, GNNEnKFStateEstimator
from wdn_kalman.gnn.manager import GNNManager
from wdn_kalman.gnn.gnn_state_estimation import load_gnn_state_estimation_data
from wdn_kalman.paths import ProjectPaths
from wdn_kalman.results import ExperimentResult


class GNNEnKFExperiment:
    """Run EnKF state estimation with the trained GNN."""

    def __init__(self, paths: ProjectPaths, gnn_manager: GNNManager):
        self.paths = paths
        self.gnn_manager = gnn_manager

    def run(
        self,
        dataset: Dataset,
        n_iters: int,
        seed: int = 0,
        buffer_size: int = 5,
        unroll_steps: int = 20,
        ensemble_size: int = 50,
        initial_variance: float = 0.01,
        process_variance: float = 0.001,
        measurement_variance: float = 0.003,
        max_steps: int | None = None,
    ) -> ExperimentResult:
        """Run the GNN EnKF for all sensor counts and placements."""
        if n_iters < 1:
            raise ValueError("n_iters must be at least 1.")

        if ensemble_size < 2:
            raise ValueError("ensemble_size must be at least 2.")

        random.seed(seed)
        np.random.seed(seed)

        data = load_gnn_state_estimation_data(
            gnn_manager=self.gnn_manager,
            dataset=dataset,
            buffer_size=buffer_size,
            unroll_steps=unroll_steps,
        )

        mean_scores = []
        std_scores = []

        print(
            f"Running {dataset.net_name} GNN EnKF: {len(dataset.sensors)} sensor counts × "
            f"{n_iters} sensor placements"
        )

        for n_sensors in dataset.sensors:
            placement_scores = []

            for _ in range(n_iters):
                config = GNNEnKFConfig(
                    num_node_quality_sensors=n_sensors,
                    num_link_sensors=n_sensors,
                    ensemble_size=ensemble_size,
                    initial_variance=initial_variance,
                    process_variance=process_variance,
                    measurement_variance=measurement_variance,
                    seed=seed,
                )

                estimator = GNNEnKFStateEstimator(data=data, config=config)
                result = estimator.run(max_steps=max_steps)
                placement_scores.append(np.asarray(result.chlorine_scores))

            combined_scores = np.concatenate([scores.reshape(-1) for scores in placement_scores])
            mean_score = float(np.mean(combined_scores))
            std_score = float(np.std(combined_scores))
            mean_scores.append(mean_score)
            std_scores.append(std_score)

            print(f"{n_sensors} sensors/type: {mean_score:.4f} ± {std_score:.4f}")

        output_file = self.paths.aggregated_results_dir / f"{dataset.file_prefix}_gnn_enkf_n_iters={n_iters}_ensemble_size={ensemble_size}_seed={seed}.npz"

        return ExperimentResult(
            dataset=dataset,
            sensors=dataset.sensors,
            mean_scores=np.asarray(mean_scores),
            std_scores=np.asarray(std_scores),
            n_iters=n_iters,
            seed=seed,
            result_file=output_file,
            ensemble_size=ensemble_size,
        )
