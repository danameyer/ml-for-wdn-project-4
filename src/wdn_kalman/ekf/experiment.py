"""Extended Kalman filter experiment."""

import random

import numpy as np

from wdn_kalman.baseline_repository import (
    load_baseline_module,
)
from wdn_kalman.datasets import Dataset
from wdn_kalman.paths import ProjectPaths
from wdn_kalman.results import ExperimentResult
from wdn_kalman.surrogate.manager import (
    SurrogateManager,
)


class EKFExperiment:
    """Run EKF state-estimation experiment."""

    def __init__(
        self,
        paths: ProjectPaths,
        surrogate_manager: SurrogateManager,
    ):
        self.paths = paths
        self.surrogates = surrogate_manager

    def run(
        self,
        dataset: Dataset,
        n_iters: int,
        seed: int = 0,
    ) -> ExperimentResult:
        """Run all sensor counts and placements."""
        if n_iters < 1:
            raise ValueError(
                "n_iters must be at least 1."
            )

        self.surrogates.validate(dataset)

        baseline_module = load_baseline_module(
            self.paths,
            "run_exp_state_estimation",
        )
        run_exp = baseline_module.run_exp

        random.seed(seed)
        np.random.seed(seed)

        mean_scores, std_scores = run_exp(
            n_sensors_range=dataset.sensors,
            n_iters=n_iters,
            net_desc=dataset.net_name,
            scada_file_in=str(
                self.surrogates.test_scada_file(
                    dataset
                )
            ),
            control_actions_file_in=str(
                self.surrogates.test_actions_file(
                    dataset
                )
            ),
            state_transition_model_file_in=str(
                self.surrogates.surrogate_file(
                    dataset
                )
            ),
        )

        output_file = (
            self.paths.aggregated_results_dir
            / (
                f"{dataset.file_prefix}_ekf_"
                f"n_iters={n_iters}_seed={seed}.npz"
            )
        )

        return ExperimentResult(
            dataset=dataset,
            sensors=dataset.sensors,
            mean_scores=np.asarray(
                mean_scores,
                dtype=float,
            ),
            std_scores=np.asarray(
                std_scores,
                dtype=float,
            ),
            n_iters=n_iters,
            seed=seed,
            result_file=output_file,
        )