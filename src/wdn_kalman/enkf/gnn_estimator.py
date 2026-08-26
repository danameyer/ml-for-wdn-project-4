"""EnKF estimation with the trained GNN transition model."""

from dataclasses import dataclass
import numpy as np
from wdn_kalman.enkf.filter import EnKFConfig, TimeVaryingEnsembleKalmanFilter
from wdn_kalman.gnn.gnn_state_estimation import (
    GNNStateEstimationData,
    GNNStateEstimationResult,
    create_random_sensor_placement,
)


@dataclass(frozen=True)
class GNNEnKFConfig:
    """Configuration for one GNN-based EnKF run."""

    num_node_quality_sensors: int = 2
    num_link_sensors: int = 2
    ensemble_size: int = 50
    initial_variance: float = 0.01
    process_variance: float = 0.001
    measurement_variance: float = 0.003
    seed: int = 42


class GNNEnKFStateEstimator:
    """Run the EnKF with the GNN transition wrapper."""

    def __init__(self, data: GNNStateEstimationData, config: GNNEnKFConfig):
        self.data = data
        self.config = config

        (self.sensor_matrix, self.sensor_node_indices, self.sensor_link_indices
        ) = create_random_sensor_placement(
            data=self.data,
            num_node_sensors=self.config.num_node_quality_sensors,
            num_link_sensors=self.config.num_link_sensors
        )

    def _predict_scaled_state(self, scaled_state: np.ndarray) -> np.ndarray:
        """Run the GNN transition model from a standardized state."""
        physical_state = self.data.inverse_scale_state(scaled_state)
        next_physical_state = self.data.wrapper.predict_with_numpy_array(physical_state)

        return self.data.scale_state(next_physical_state)

    def run(self, max_steps: int | None = None) -> GNNStateEstimationResult:
        """Run the EnKF over the GNN test sequence."""
        enkf = self._create_filter()
        available_steps = len(self.data.node_concentrations) - 1

        if max_steps is None:
            num_steps = available_steps
        else:
            num_steps = min(max_steps,available_steps)

        scores = []
        node_predictions = []
        node_true_values = []
        node_uncertainties = []
        link_predictions = []
        link_true_values = []

        for target_index in range(1, num_steps + 1):
            input_index = target_index - 1
            source_indices, source_values = (self.data.source_state_values(time_index=input_index))

            flow_indices, flow_values = (
                self.data.flow_sensor_state_values(
                    time_index=target_index,
                    sensor_link_indices=self.sensor_link_indices,
                )
            )

            enkf.set_exact_state_values(
                indices=source_indices,
                values=self.data.scale_state_values(indices=source_indices, values=source_values)
            )

            enkf.set_state_values(
                indices=flow_indices,
                values=self.data.scale_state_values(indices=flow_indices, values=flow_values)
            )

            self.sensor_matrix = (
                self.data.create_sensor_matrix(
                    time_index=target_index,
                    sensor_node_indices=self.sensor_node_indices,
                    sensor_link_indices=self.sensor_link_indices,
                )
            )

            observation = (
                self.data.create_scaled_observation(
                    time_index=target_index,
                    sensor_node_indices=self.sensor_node_indices,
                    sensor_link_indices=self.sensor_link_indices,
                )
            )

            enkf.step(np.asarray(observation, dtype=float))
            ensemble_scaled = enkf.ensemble
            ensemble_physical = self.data.inverse_scale_state(ensemble_scaled)
            estimated_state = ensemble_physical.mean(axis=0)
            estimated_std = ensemble_physical.std(axis=0, ddof=1)
            predicted_nodes = estimated_state[self.data.wrapper.node_slice]
            actual_nodes = self.data.node_concentrations[target_index]
            node_std = estimated_std[self.data.wrapper.node_slice]
            predicted_links = self.data.link_concentration_estimates(state=estimated_state, time_index=target_index)
            actual_links = self.data.link_concentrations[target_index]
            predicted_chlorine = np.concatenate((predicted_nodes, predicted_links))
            actual_chlorine = np.concatenate((actual_nodes, actual_links))
            scores.append(np.median(np.abs(predicted_chlorine - actual_chlorine)))
            node_predictions.append(predicted_nodes)
            node_true_values.append(actual_nodes)
            node_uncertainties.append(node_std)
            link_predictions.append(predicted_links)
            link_true_values.append(actual_links)

        return GNNStateEstimationResult(
            dataset=self.data.dataset,
            chlorine_scores=np.asarray(scores),
            node_chlorine_predictions=np.vstack(node_predictions),
            node_chlorine_true=np.vstack(node_true_values),
            node_chlorine_std=np.vstack(node_uncertainties),
            link_chlorine_predictions=np.vstack(link_predictions),
            link_chlorine_true=np.vstack(link_true_values),
            sensor_matrix=self.sensor_matrix.copy(),
            sensor_node_indices=self.sensor_node_indices.copy(),
            sensor_link_indices=self.sensor_link_indices.copy(),
        )

    def _create_filter(self) -> TimeVaryingEnsembleKalmanFilter:
        """Create the custom time-varying EnKF."""
        state_dim = self.data.state_dim
        observation_dim = self.sensor_matrix.shape[0]

        def get_measurement_function(_time_step):
            return lambda state: self.sensor_matrix@ np.asarray(state).reshape(-1)

        def get_transition_function(_time_step):
            return self._predict_scaled_state

        return TimeVaryingEnsembleKalmanFilter(
            state_dim=state_dim,
            obs_dim=observation_dim,
            init_state=self.data.scale_state(self.data.create_initial_state()),
            get_state_transition_func=get_transition_function,
            get_measurement_func=get_measurement_function,
            init_state_uncertainty_cov=self.config.initial_variance * np.eye(state_dim),
            system_uncertainty_cov=self.config.process_variance * np.eye(state_dim),
            measurement_uncertainty_cov=self.config.measurement_variance * np.eye(observation_dim),
            config=EnKFConfig(ensemble_size=self.config.ensemble_size, seed=self.config.seed)
        )
