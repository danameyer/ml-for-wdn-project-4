"""Preparation for GNN-based state estimation experiments."""

from dataclasses import dataclass
import random
import numpy as np
import torch
from epyt_flow.simulation import ScadaData
from wdn_kalman.datasets import Dataset
from wdn_kalman.gnn.data import prepare_gnn_data
from wdn_kalman.gnn.manager import GNNManager
from wdn_kalman.gnn.wrapper import GNNTransitionWrapper


@dataclass
class GNNStateEstimationData:
    """GNN model and test sequence used by Kalman filters."""

    dataset: Dataset
    wrapper: GNNTransitionWrapper
    node_concentrations: np.ndarray
    link_concentrations: np.ndarray
    edge_flows: np.ndarray
    source_node_indices: np.ndarray
    non_source_node_indices: np.ndarray
    directed_link_columns: np.ndarray
    pipe_link_indices: np.ndarray
    state_mean: np.ndarray
    state_scale: np.ndarray
    link_quality_scale: np.ndarray

    @property
    def state_dim(self) -> int:
        return self.wrapper.state_dim

    @property
    def num_nodes(self) -> int:
        return self.wrapper.num_nodes

    @property
    def num_edges(self) -> int:
        return self.wrapper.num_edges

    @property
    def flow_state_indices(self) -> np.ndarray:
        return np.arange(self.wrapper.flow_start, self.wrapper.flow_end, dtype=int)

    def scale_state(self, state: np.ndarray) -> np.ndarray:
        """Convert physical units to standardized coordinates."""
        state = np.asarray(state, dtype=float)

        return (state - self.state_mean) / self.state_scale

    def inverse_scale_state(self, state: np.ndarray) -> np.ndarray:
        """Convert standardized filter coordinates back to physical units."""
        state = np.asarray(state, dtype=float)

        return state * self.state_scale + self.state_mean

    def scale_state_values(self, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Scale selected physical unit values."""
        indices = np.asarray(indices, dtype=int)
        values = np.asarray(values, dtype=float)

        return (values - self.state_mean[indices]) / self.state_scale[indices]

    def directed_edges_for_link(self,link_index: int) -> np.ndarray:
        """Return the two directed GNN edges for one physical link."""
        return np.flatnonzero(self.directed_link_columns == int(link_index)).astype(int)

    def return_directed_edge(self, time_index: int, link_index: int) -> int | None:
        """Return the directed GNN edge for one physical link."""
        directed_edges = (self.directed_edges_for_link(link_index))
        physical_flow = float(self.edge_flows[time_index, link_index])

        if abs(physical_flow) < 1e-12:
            return None

        directed_flow_signs = self.wrapper.directed_flow_signs[directed_edges].detach().cpu().numpy()
        directed_flows = physical_flow * directed_flow_signs

        return int(directed_edges[np.argmax(directed_flows)])

    def create_initial_state(self) -> np.ndarray:
        """Create the initial flat GNN state."""
        node_concentrations = torch.as_tensor(
            self.node_concentrations[0],
            dtype=torch.float32,
            device=self.wrapper.edge_attr.device,
        )

        link_flows = torch.as_tensor(
            self.edge_flows[1],
            dtype=torch.float32,
            device=self.wrapper.edge_attr.device,
        )

        initial_state = (
            self.wrapper.create_initial_state(node_concentrations=node_concentrations,
                                              link_flows=link_flows
                                              )
        )

        return initial_state.detach().cpu().numpy()

    def source_state_values(self, time_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return known source chlorine entries."""
        values = self.node_concentrations[time_index, self.source_node_indices]

        return self.source_node_indices.copy(), values.copy()

    def flow_sensor_state_values(
            self,
            time_index: int,
            sensor_link_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return physical flow state entries."""
        sensor_link_indices = np.asarray(sensor_link_indices, dtype=int)
        state_indices = self.wrapper.flow_start + sensor_link_indices
        values = self.edge_flows[time_index, sensor_link_indices]

        return state_indices.astype(int), np.asarray(values, dtype=float)

    def create_observation(
            self,
            time_index: int,
            sensor_node_indices: np.ndarray,
            sensor_link_indices: np.ndarray,
    ) -> np.ndarray:
        """Create node chlorine, link chlorine and flow observations."""
        node_observations = self.node_concentrations[time_index, sensor_node_indices]
        link_observations = self.link_concentrations[time_index, sensor_link_indices]
        flow_observations = self.edge_flows[time_index, sensor_link_indices]

        return np.concatenate((node_observations, link_observations, flow_observations)).astype(float)

    def create_scaled_observation(
            self,
            time_index: int,
            sensor_node_indices: np.ndarray,
            sensor_link_indices: np.ndarray,
    ) -> np.ndarray:
        """Create the complete observation in filter coordinates."""
        sensor_node_indices = np.asarray(sensor_node_indices, dtype=int)
        sensor_link_indices = np.asarray(sensor_link_indices, dtype=int)

        # Node chlorine
        node_values = self.node_concentrations[time_index, sensor_node_indices]
        node_observations = self.scale_state_values(indices=sensor_node_indices, values=node_values)

        # Link chlorine
        link_observations = (
                self.link_concentrations[time_index, sensor_link_indices]
                / self.link_quality_scale[sensor_link_indices]
        )

        # Physical link flow
        flow_state_indices = self.wrapper.flow_start + sensor_link_indices
        flow_values = self.edge_flows[time_index, sensor_link_indices]
        flow_observations = self.scale_state_values(indices=flow_state_indices, values=flow_values)

        return np.concatenate((node_observations, link_observations, flow_observations))

    def create_sensor_matrix(
            self,
            time_index: int,
            sensor_node_indices: np.ndarray,
            sensor_link_indices: np.ndarray
    ) -> np.ndarray:
        """Create sensor matrix for node chlorine, link chlorine, and link flow."""
        sensor_node_indices = np.asarray(sensor_node_indices, dtype=int)
        sensor_link_indices = np.asarray(sensor_link_indices, dtype=int)
        num_node_sensors = len(sensor_node_indices)
        num_link_sensors = len(sensor_link_indices)
        observation_dim = num_node_sensors + 2 * num_link_sensors
        sensor_matrix = np.zeros((observation_dim, self.state_dim), dtype=float,)

        # Node chlorine
        for row, node_index in enumerate(sensor_node_indices):
            sensor_matrix[row, int(node_index)] = 1.0

        # Link chlorine
        link_row_start = num_node_sensors

        for offset, link_index in enumerate(sensor_link_indices):
            row = link_row_start + offset
            active_edge = self.return_directed_edge(time_index=time_index, link_index=int(link_index))
            link_scale = self.link_quality_scale[int(link_index)]

            if active_edge is not None:
                buffer_start = self.wrapper.buffer_start + active_edge * self.wrapper.buffer_size
                buffer_end = buffer_start + self.wrapper.buffer_size
                sensor_matrix[row, buffer_start:buffer_end] = 1.0 / (self.wrapper.buffer_size * link_scale)

            else:
                # Flow direction is undefined when q == 0.
                directed_edges = self.directed_edges_for_link(int(link_index))
                weight = 1.0 / (len(directed_edges) * self.wrapper.buffer_size * link_scale)

                for directed_edge in directed_edges:
                    buffer_start = self.wrapper.buffer_start + int(directed_edge) * self.wrapper.buffer_size
                    buffer_end = buffer_start + self.wrapper.buffer_size
                    sensor_matrix[row, buffer_start:buffer_end] = weight

        # Link flow

        flow_row_start = num_node_sensors + num_link_sensors

        for offset, link_index in enumerate(sensor_link_indices):
            row = flow_row_start + offset
            flow_state_index = self.wrapper.flow_start + int(link_index)
            sensor_matrix[row, flow_state_index] = 1.0

        return sensor_matrix

    def link_concentration_estimates(self, state: np.ndarray, time_index: int) -> np.ndarray:
        """Derive physical link chlorine estimates from GNN buffer."""
        state = np.asarray(state, dtype=float)
        edge_buffer = state[self.wrapper.buffer_slice].reshape(self.wrapper.num_edges, self.wrapper.buffer_size)
        estimates = np.empty(self.wrapper.num_links, dtype=float)

        for link_index in range(self.wrapper.num_links):
            directed_edge = self.return_directed_edge(time_index=time_index, link_index=link_index)

            if directed_edge is not None:
                estimates[link_index] = np.mean(edge_buffer[directed_edge])

            else:
                directed_edges = self.directed_edges_for_link(link_index)
                estimates[link_index] = np.mean(edge_buffer[directed_edges])

        return estimates


@dataclass
class GNNStateEstimationResult:
    """Result of one GNN state estimation run."""

    dataset: Dataset
    chlorine_scores: np.ndarray
    node_chlorine_predictions: np.ndarray
    node_chlorine_true: np.ndarray
    node_chlorine_std: np.ndarray
    sensor_matrix: np.ndarray
    sensor_node_indices: np.ndarray
    sensor_link_indices: np.ndarray
    link_chlorine_predictions: np.ndarray
    link_chlorine_true: np.ndarray

    @property
    def num_steps(self) -> int:
        return len(self.chlorine_scores)

    @property
    def mean_chlorine_score(self) -> float:
        return float(np.mean(self.chlorine_scores))


def _prepare_directed_link_mapping(scada_data: ScadaData, edge_index: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Map directed GNN edges to physical network links."""
    topology = scada_data.get_attributes()["network_topo"]
    idx_to_node = {index: node_id for index, node_id in enumerate(topology.nodes)}
    physical_links = list(topology.get_all_links())
    link_id_to_column = {str(link_id): column for column, (link_id, _) in enumerate(physical_links)}
    directed_link_columns = []
    pipe_link_indices = set()

    for edge in zip(edge_index[0], edge_index[1]):
        source_index = int(edge[0])
        target_index = int(edge[1])
        source_node = idx_to_node[source_index]
        target_node = idx_to_node[target_index]
        info = topology.get_edge_data(source_node, target_node)["info"]
        link_column = link_id_to_column[str(info["id"])]
        directed_link_columns.append(link_column)

        # Type 1 = ordinary pipe.
        if info["type"] == 1:
            pipe_link_indices.add(link_column)

    return np.asarray(directed_link_columns, dtype=int), np.asarray(sorted(pipe_link_indices), dtype=int)


def load_gnn_state_estimation_data(
    gnn_manager: GNNManager,
    dataset: Dataset,
    buffer_size: int,
    unroll_steps: int,
) -> GNNStateEstimationData:
    """Load the trained GNN and the test sequence used for estimation."""
    test_scada_file = gnn_manager.paths.data_dir / f"{dataset.file_prefix}_randDemand=False_test.epytflow_scada_data"
    training_scada_file = gnn_manager.paths.data_dir / f"{dataset.file_prefix}_randDemand=True_training.epytflow_scada_data"
    model_file = gnn_manager.gnn_file(dataset=dataset, buffer_size=buffer_size, unroll_steps=unroll_steps)

    if not test_scada_file.exists():
        raise FileNotFoundError(test_scada_file)

    if not training_scada_file.exists():
        raise FileNotFoundError(training_scada_file)

    if not model_file.exists():
        raise FileNotFoundError(model_file)

    scada_data = ScadaData.load_from_file(str(test_scada_file))
    training_scada_data = (ScadaData.load_from_file(str(training_scada_file)))
    link_concentrations = np.asarray(scada_data.get_data_links_quality(), dtype=float,)
    training_link_concentrations = np.asarray(training_scada_data.get_data_links_quality(), dtype=float)

    (
        device,
        edge_attr,
        edge_flows,
        edge_index,
        node_concentrations_matrix,
        non_source_nodes_mask,
    ) = prepare_gnn_data(scada_data)

    (
        _,
        _,
        training_edge_flows,
        training_edge_index,
        training_node_concentrations,
        _,
    ) = prepare_gnn_data(training_scada_data)

    if not torch.equal(training_edge_index, edge_index):
        raise ValueError("Training and test datasets have different GNN edge ordering.")

    model, checkpoint = gnn_manager.load(
        dataset=dataset,
        buffer_size=buffer_size,
        unroll_steps=unroll_steps,
        device=device,
    )

    checkpoint_edge_index = checkpoint["edge_index"].to(device=device, dtype=torch.long)
    checkpoint_edge_attr = checkpoint["edge_attr"].to(device=device, dtype=torch.float32)

    if not torch.equal(checkpoint_edge_index, edge_index):
        raise ValueError("The GNN checkpoint topology does not match the test dataset topology.")

    if checkpoint_edge_attr.shape != edge_attr.shape:
        raise ValueError("The GNN checkpoint edge attributes and test edge attributes have different shapes.")

    (
        directed_link_columns, pipe_link_indices
    ) = _prepare_directed_link_mapping(scada_data=scada_data, edge_index=edge_index)

    wrapper = GNNTransitionWrapper(
        model=model,
        edge_index=checkpoint_edge_index,
        edge_attr=edge_attr,
        directed_link_columns=directed_link_columns,
    ).to(device)

    wrapper.eval()

    state_mean = np.zeros(wrapper.state_dim, dtype=float)
    state_scale = np.ones(wrapper.state_dim, dtype=float)
    training_node_values = training_node_concentrations.squeeze(-1).detach().cpu().numpy()
    training_link_flows = wrapper.collapse_edge_flows(training_edge_flows)
    training_flow_values = training_link_flows.detach().cpu().numpy()
    node_mean = np.mean(training_node_values, axis=0)
    node_scale = np.std(training_node_values, axis=0)
    flow_mean = np.mean(training_flow_values, axis=0)
    flow_scale = np.std(training_flow_values, axis=0)
    minimum_scale = 1e-8
    link_quality_scale = np.std(training_link_concentrations, axis=0)
    link_quality_scale = np.where(link_quality_scale > minimum_scale, link_quality_scale,1.0)
    node_scale = np.where(node_scale > minimum_scale, node_scale,1.0,)
    flow_scale = np.where(flow_scale > minimum_scale, flow_scale,1.0,)
    state_mean[wrapper.node_slice] = node_mean
    state_scale[wrapper.node_slice] = node_scale
    state_mean[wrapper.flow_slice] = flow_mean
    state_scale[wrapper.flow_slice] = flow_scale
    non_source_mask = non_source_nodes_mask.reshape(-1).detach().cpu().numpy().astype(bool)
    non_source_node_indices = np.flatnonzero(non_source_mask).astype(int)
    source_node_indices = np.flatnonzero(~non_source_mask).astype(int)
    node_concentrations = node_concentrations_matrix.squeeze(-1).detach().cpu().numpy()
    physical_link_flows = wrapper.collapse_edge_flows(edge_flows)
    edge_flows_array = physical_link_flows.detach().cpu().numpy()

    return GNNStateEstimationData(
        dataset=dataset,
        wrapper=wrapper,
        node_concentrations=node_concentrations,
        link_concentrations=link_concentrations,
        edge_flows=edge_flows_array,
        source_node_indices=source_node_indices,
        non_source_node_indices=non_source_node_indices,
        directed_link_columns=directed_link_columns,
        pipe_link_indices=pipe_link_indices,
        state_mean=state_mean,
        state_scale=state_scale,
        link_quality_scale=link_quality_scale,
    )

def create_random_sensor_placement(
        data: GNNStateEstimationData,
        num_node_sensors: int,
        num_link_sensors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create random node-chlorine and physical-link flow sensors."""
    if num_node_sensors < 1:
        raise ValueError("num_node_sensors must be at least 1.")

    if num_link_sensors < 1:
        raise ValueError("num_link_sensors must be at least 1.")

    available_nodes = data.non_source_node_indices.tolist()
    available_links = data.pipe_link_indices.tolist()

    if num_node_sensors > len(available_nodes):
        raise ValueError("Cannot place more node chlorine sensors than available non-source nodes.")

    if num_link_sensors > len(available_links):
        raise ValueError("Cannot place more link sensors than available pipes.")

    sensor_node_indices = np.asarray(random.sample(available_nodes, k=num_node_sensors), dtype=int)
    sensor_node_indices.sort()

    sensor_link_indices = np.asarray(random.sample(available_links, k=num_link_sensors), dtype=int)
    sensor_link_indices.sort()

    sensor_matrix = (
        data.create_sensor_matrix(time_index=1, sensor_node_indices=sensor_node_indices, sensor_link_indices=sensor_link_indices)
    )

    return sensor_matrix, sensor_node_indices, sensor_link_indices
