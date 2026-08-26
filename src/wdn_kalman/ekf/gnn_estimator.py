"""Using EKF with the trained GNN transition model."""

from dataclasses import dataclass
import numpy as np
import torch
from epyt_control.signal_processing.state_estimation import TimeVaryingExtendedKalmanFilter
from wdn_kalman.gnn.gnn_state_estimation import (
    GNNStateEstimationData,
    GNNStateEstimationResult,
    create_random_sensor_placement
)


@dataclass(frozen=True)
class GNNEKFConfig:
    """Config for GNN-based EKF run."""
    num_node_sensors: int = 2
    num_link_sensors: int = 2
    initial_variance: float = 0.01
    process_variance: float = 0.001
    measurement_variance: float = 0.01


class GNNEKFStateEstimator:
    """Run EKF with the GNN transition wrapper."""

    def __init__(self, data: GNNStateEstimationData, config: GNNEKFConfig):
        self.data = data
        self.config = config

        (
            self.sensor_matrix,
            self.sensor_node_indices,
            self.sensor_link_indices,
        ) = create_random_sensor_placement(
            data=self.data,
            num_node_sensors=self.config.num_node_sensors,
            num_link_sensors=self.config.num_link_sensors
        )

    def run(self, max_steps: int | None = None) -> GNNStateEstimationResult:
        """Run the EKF over the GNN test sequence."""
        ekf = self._create_filter()
        available_steps = len(self.data.node_concentrations) - 1

        if max_steps is None:
            num_steps = available_steps
        else:
            num_steps = min(max_steps, available_steps)

        scores = []
        node_predictions = []
        node_true_values = []
        node_uncertainties = []
        link_predictions = []
        link_true_values = []

        for target_index in range(1, num_steps + 1):
            input_index = target_index - 1

            self._inject_known_inputs(ekf=ekf,
                                      source_time_index=input_index,
                                      flow_time_index=target_index
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

            estimated_state_scaled, covariance_scaled = ekf.step(np.asarray(observation, dtype=float))
            estimated_state = self.data.inverse_scale_state(estimated_state_scaled)
            state_std_scaled = np.sqrt(np.maximum(np.diag(covariance_scaled), 0.0))
            state_std = state_std_scaled * self.data.state_scale
            predicted_nodes = (estimated_state[self.data.wrapper.node_slice])
            actual_nodes = self.data.node_concentrations[target_index]
            node_std = state_std[self.data.wrapper.node_slice]
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
            sensor_link_indices=self.sensor_link_indices.copy()
        )

    def _create_filter(
        self,
    ) -> TimeVaryingExtendedKalmanFilter:
        """Create the time-varying EKF (based on epyt-flow design)."""
        state_dim = self.data.state_dim
        observation_dim = self.sensor_matrix.shape[0]

        def get_measurement_function(_time_step):
            return lambda state: self.sensor_matrix @ np.asarray(state).reshape(-1)

        def get_measurement_jacobian(_time_step):
            return lambda _state: self.sensor_matrix

        def get_transition_function(_time_step):
            return self._predict_scaled_state

        def get_transition_jacobian(_time_step):
            return self._compute_jacobian

        return TimeVaryingExtendedKalmanFilter(
            state_dim=state_dim,
            obs_dim=observation_dim,
            init_state=self.data.scale_state(self.data.create_initial_state()),
            get_state_transition_func=get_transition_function,
            get_state_transition_func_grad=get_transition_jacobian,
            get_measurement_func=get_measurement_function,
            get_measurement_func_grad=get_measurement_jacobian,
            init_state_uncertainty_cov=self.config.initial_variance * np.eye(state_dim),
            system_uncertainty_cov=self.config.process_variance * np.eye(state_dim),
            measurement_uncertainty_cov=self.config.measurement_variance * np.eye(observation_dim)
        )

    def _compute_jacobian(self, state: np.ndarray) -> np.ndarray:
        """Compute the Jacobian of the GNN transition model."""
        device = self.data.wrapper.edge_attr.device
        state_tensor = torch.as_tensor(
            np.asarray(state, dtype=float,).reshape(-1),
            dtype=torch.float32,
            device=device,
        ).clone().detach().requires_grad_(True)

        state_mean = torch.as_tensor(self.data.state_mean, dtype=torch.float32, device=device)
        state_scale = torch.as_tensor(self.data.state_scale, dtype=torch.float32, device=device)

        def scale_transition(scaled_state: torch.Tensor) -> torch.Tensor:
            physical_state = scaled_state  * state_scale + state_mean
            next_physical_state = self.data.wrapper(physical_state)

            return (next_physical_state - state_mean) / state_scale

        with torch.enable_grad():
            jacobian = torch.autograd.functional.jacobian(scale_transition, state_tensor, create_graph=False, vectorize=True)

        return jacobian.detach().cpu().numpy()

    def _inject_known_inputs(
            self,
            ekf: TimeVaryingExtendedKalmanFilter,
            source_time_index: int,
            flow_time_index: int,
    ) -> None:
        """Insert source chlorine and measured flows."""
        (source_indices, source_values) = self.data.source_state_values(
            time_index=source_time_index
        )

        (flow_indices, flow_values) = self.data.flow_sensor_state_values(
            time_index=flow_time_index,
            sensor_link_indices=self.sensor_link_indices
        )

        ekf._x[source_indices] = self.data.scale_state_values(indices=source_indices, values=source_values)

        ekf._x[flow_indices] = self.data.scale_state_values(indices=flow_indices, values=flow_values)

    def _predict_scaled_state(self, scaled_state: np.ndarray) -> np.ndarray:
        """Run the GNN transition model from a standardized state."""
        physical_state = self.data.inverse_scale_state(scaled_state)
        next_physical_state = self.data.wrapper.predict_with_numpy_array(physical_state)

        return self.data.scale_state(next_physical_state)