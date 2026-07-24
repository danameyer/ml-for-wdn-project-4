"""Run Ensemble Kalman Filter."""

from dataclasses import dataclass

import numpy as np
from epyt_flow.simulation import ScadaData

from wdn_kalman.baseline_repository import load_baseline_module
from wdn_kalman.datasets import Dataset
from wdn_kalman.enkf.filter import (
    EnKFConfig,
    TimeVaryingEnsembleKalmanFilter,
)
from wdn_kalman.surrogate.manager import SurrogateManager


@dataclass(frozen=True)
class EnKFExperimentConfig:
    num_node_quality_sensors: int = 2
    num_link_sensors: int = 2
    ensemble_size: int = 50
    initial_variance: float = 0.01
    process_variance: float = 0.001
    measurement_variance: float = 0.01
    seed: int = 42


@dataclass
class EnKFStateEstimationResult:
    dataset: Dataset
    chlorine_scores: np.ndarray
    chlorine_predictions: np.ndarray
    chlorine_true: np.ndarray
    chlorine_std: np.ndarray
    sensor_matrix: np.ndarray
    flow_indices: np.ndarray

    @property
    def num_steps(self):
        return len(self.chlorine_scores)

    @property
    def mean_chlorine_score(self):
        return float(
            np.mean(self.chlorine_scores)
        )


class EnKFStateEstimator:
    """
    Run the custom EnKF with Scada data and the trained neural surrogate model.
    """

    def __init__(
            self,
            surrogate_manager: SurrogateManager,
            dataset: Dataset,
            config: EnKFExperimentConfig,
    ):
        self.surrogates = surrogate_manager
        self.dataset = dataset
        self.config = config

    def run(self, max_steps=None):
        """Run EnKF for given time sequence."""
        self._prepare_repository_data()
        enkf = self._create_filter()
        available_steps = (len(self.states_scaled) - 1)

        if max_steps is None:
            num_steps = available_steps
        else:
            num_steps = min(
                max_steps,
                available_steps,
            )

        scores = []
        predictions = []
        true_values = []
        uncertainties = []

        for target_index in range(1, num_steps + 1):
            true_state_scaled = (self.states_scaled[target_index])

            enkf.set_state_values(indices=self.flow_indices,
                                  values=true_state_scaled[self.flow_indices]
                                  )

            observation = (self.sensor_matrix @ true_state_scaled)

            enkf.step(observation)

            ensemble_physical = (self.convert_to_physical_units(enkf.ensemble))

            estimated_state = (ensemble_physical.mean(axis=0))

            estimated_std = (ensemble_physical.std(axis=0, ddof=1))

            true_state = self.convert_to_physical_units(true_state_scaled)[0]

            predicted_chlorine = (estimated_state[:self.num_chlorine_states])

            actual_chlorine = (true_state[:self.num_chlorine_states])

            predictions.append(predicted_chlorine)

            true_values.append(actual_chlorine)

            uncertainties.append(estimated_std[:self.num_chlorine_states])

            scores.append(np.median(np.abs(predicted_chlorine - actual_chlorine)))

        return EnKFStateEstimationResult(
            dataset=self.dataset,
            chlorine_scores=np.asarray(scores),
            chlorine_predictions=np.vstack(predictions),
            chlorine_true=np.vstack(true_values),
            chlorine_std=np.vstack(uncertainties),
            sensor_matrix=(self.sensor_matrix.copy()),
            flow_indices=(self.flow_indices.copy())
        )

    def _prepare_repository_data(self):
        """Load and prepare the repository data."""
        baseline_module = load_baseline_module(
            self.surrogates.paths,
            "run_exp_state_estimation",
        )

        create_random_sensor_placement = (
            baseline_module.create_random_sensor_placement
        )
        get_state_transition_model = (
            baseline_module.get_state_transition_model
        )

        scada_file = self.surrogates.test_scada_file(
            self.dataset
        )

        control_file = self.surrogates.test_actions_file(
            self.dataset
        )

        model_file = self.surrogates.surrogate_file(
            self.dataset
        )
        scaler_file = (
            self.surrogates
            .surrogate_scaler_file(self.dataset)
        )

        self.surrogates.validate(self.dataset)

        for path in (scada_file, control_file, model_file, scaler_file):
            if not path.exists():
                raise FileNotFoundError(path)

        scada = ScadaData.load_from_file(str(scada_file))
        flows = scada.get_data_flows()
        node_quality = (scada.get_data_nodes_quality())
        link_quality = (scada.get_data_links_quality())

        with np.load(control_file) as data:
            control_actions = data["control_actions"]

        states_physical = np.concatenate((node_quality[:-1], link_quality[:-1], flows[1:]), axis=1,)
        self.controls = control_actions[:len(states_physical)]
        self.num_nodes = node_quality.shape[1]
        self.num_links = link_quality.shape[1]
        self.num_chlorine_states = (self.num_nodes + self.num_links)
        self.state_dim = (states_physical.shape[1])
        self.model = get_state_transition_model(self.dataset.net_name, str(model_file))
        self.model.n_missing_flows = self.num_chlorine_states
        self.model._normalize_input_output = False

        state_and_control = np.concatenate((states_physical, self.controls), axis=1)

        self.states_scaled = self.model._scaler.transform(state_and_control)[:, :self.state_dim]

        (self.sensor_matrix, flow_indices) = create_random_sensor_placement(
            n_node_quality_sensors=(
                self.config
                .num_node_quality_sensors
            ),
            n_link_sensors=(
                self.config.num_link_sensors
            ),
            n_nodes=self.num_nodes,
            n_links=self.num_links,
            state_dim=self.state_dim,
        )

        self.flow_indices = np.asarray(flow_indices, dtype=int)
        self.observation_dim = (self.sensor_matrix.shape[0])

    def _create_filter(self):
        """Create EnKF filter for run."""
        def get_measurement_function(_time_step):
            return lambda state: (
                self.sensor_matrix @ np.asarray(state).reshape(-1)
            )

        def get_transition_function(time_step):
            control = self.controls[time_step + 1].reshape(1, -1)

            def transition(state):
                return self.model.predict(np.asarray(state).reshape(1,-1), control).flatten()

            return transition

        return TimeVaryingEnsembleKalmanFilter(
            state_dim=self.state_dim,
            obs_dim=self.observation_dim,
            init_state=self.states_scaled[0],
            get_state_transition_func=get_transition_function,
            get_measurement_func=get_measurement_function,
            init_state_uncertainty_cov=(self.config.initial_variance * np.eye(self.state_dim)),
            system_uncertainty_cov=(self.config.process_variance * np.eye(self.state_dim)),
            measurement_uncertainty_cov=(self.config.measurement_variance * np.eye(self.observation_dim)),
            config=EnKFConfig(ensemble_size=self.config.ensemble_size, seed=self.config.seed))

    def convert_to_physical_units(
        self,
        scaled_states,
    ):
        """Convert scaled states back to physical units."""
        scaled_states = np.atleast_2d(scaled_states)
        control_padding = np.zeros((len(scaled_states), self.controls.shape[1]))
        padded_states = np.concatenate((scaled_states, control_padding), axis=1)

        return (
            self.model._scaler.inverse_transform(padded_states)[:, :self.state_dim]
        )