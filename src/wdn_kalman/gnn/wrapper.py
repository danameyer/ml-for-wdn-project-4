"""Transition wrapper for the GNN (needed for Kalman filter compatibility)."""

import numpy as np
import torch
import torch.nn as nn
from wdn_kalman.gnn.model import GNN


class GNNTransitionWrapper(nn.Module):
    """Wrap the GNN as a flat state transition model."""

    def __init__(
        self,
        model: GNN,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        directed_link_columns: np.ndarray,
    ):
        super().__init__()
        self.model = model
        self.model.eval()
        self.buffer_size = model.buffer_size

        self.register_buffer("edge_index", edge_index.to(dtype=torch.long))
        self.register_buffer("edge_attr", edge_attr.to(dtype=torch.float32))

        self.num_edges = self.edge_index.size(1)
        self.num_nodes = int(self.edge_index.max().item()) + 1

        directed_link_columns = torch.as_tensor(
            directed_link_columns,
            dtype=torch.long,
            device=self.edge_index.device,
        )

        if directed_link_columns.shape != (self.num_edges,):
            raise ValueError(
                "directed_link_columns must contain one physical-link index per directed GNN edge."
            )

        self.register_buffer("directed_link_columns", directed_link_columns)

        self.num_links = int(directed_link_columns.max().item()) + 1

        (
            directed_flow_signs,
            representative_edge_indices,
        ) = self._create_flow_direction_mapping(directed_link_columns)

        self.register_buffer("directed_flow_signs", directed_flow_signs)
        self.register_buffer("representative_edge_indices", representative_edge_indices)

        self.node_start = 0
        self.node_end = self.num_nodes
        self.buffer_start = self.node_end
        self.buffer_end = self.buffer_start + self.num_edges * self.buffer_size
        self.flow_start = self.buffer_end
        self.flow_end = self.flow_start + self.num_links
        self.state_dim = self.flow_end

    @staticmethod
    def _create_flow_direction_mapping(directed_link_columns: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """Create mapping between one physical flow value and the two directed GNN edge-flow values."""
        num_edges = len(directed_link_columns)
        num_links = (int(directed_link_columns.max().item()) + 1)

        directed_flow_signs = torch.empty(
            num_edges,
            dtype=torch.float32,
            device=directed_link_columns.device,
        )

        representative_edge_indices = torch.empty(
            num_links,
            dtype=torch.long,
            device=directed_link_columns.device,
        )

        for link_index in range(num_links):
            directed_edges = torch.where(directed_link_columns == link_index)[0]

            if len(directed_edges) != 2:
                raise ValueError(
                    "Expected exactly two directed GNN edges for physical link "
                    f"{link_index}, but found {len(directed_edges)}."
                )

            first_edge = directed_edges[0]
            second_edge = directed_edges[1]
            representative_edge_indices[link_index] = first_edge
            directed_flow_signs[first_edge] = 1.0
            directed_flow_signs[second_edge] = -1.0

        return directed_flow_signs, representative_edge_indices,

    @property
    def node_slice(self) -> slice:
        return slice(self.node_start, self.node_end)

    @property
    def buffer_slice(self) -> slice:
        return slice(self.buffer_start, self.buffer_end)

    @property
    def flow_slice(self) -> slice:
        return slice(self.flow_start, self.flow_end)

    @staticmethod
    def pack_gnn_state_model(
        node_concentrations: torch.Tensor,
        edge_buffer: torch.Tensor,
        link_flows: torch.Tensor,
    ) -> torch.Tensor:
        """Pack the Kalman state."""
        return torch.cat(
        (node_concentrations.reshape(-1),
                edge_buffer.reshape(-1),
                link_flows.reshape(-1),
            ), dim=0,
        )

    def unpack_gnn_state_model(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack the flat Kalman state."""
        node_concentrations = state[self.node_slice].reshape(self.num_nodes, 1)
        edge_buffer = state[self.buffer_slice].reshape(self.num_edges, self.buffer_size)
        link_flows = state[self.flow_slice].reshape(self.num_links)

        return node_concentrations, edge_buffer, link_flows

    def expand_link_flows(self, link_flows: torch.Tensor) -> torch.Tensor:
        """Expand physical link flows to directed GNN flows."""
        return link_flows[self.directed_link_columns] * self.directed_flow_signs

    def collapse_edge_flows(self, edge_flows: torch.Tensor) -> torch.Tensor:
        """Convert directed GNN flow features to one flow value per physical link."""
        return edge_flows[
            ...,
            self.representative_edge_indices,
        ]

    def create_initial_state(
        self,
        node_concentrations: torch.Tensor,
        link_flows: torch.Tensor,
    ) -> torch.Tensor:
        """Create the initial physical Kalman state."""
        device = self.edge_attr.device
        node_concentrations = node_concentrations.to(device=device, dtype=torch.float32).reshape(self.num_nodes, 1)
        link_flows = link_flows.to(device=device, dtype=torch.float32).reshape(self.num_links)
        edge_buffer = torch.zeros((self.num_edges, self.buffer_size), dtype=torch.float32, device=device)

        return self.pack_gnn_state_model(
            node_concentrations=node_concentrations,
            edge_buffer=edge_buffer,
            link_flows=link_flows,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Apply one GNN transition."""
        (node_concentrations,edge_buffer, link_flows) = self.unpack_gnn_state_model(state)
        edge_flows = self.expand_link_flows(link_flows)

        dynamic_edge_attr = torch.stack(
            (
                edge_flows,
                self.edge_attr[:, 0],
                self.edge_attr[:, 1],
            ), dim=-1,
        )

        (
            next_node_concentrations,
            next_edge_buffer,
            _,
        ) = self.model(node_concentrations, self.edge_index, edge_buffer, dynamic_edge_attr)

        return self.pack_gnn_state_model(
            node_concentrations=next_node_concentrations,
            edge_buffer=next_edge_buffer,
            link_flows=link_flows,
        )

    def predict_with_numpy_array(self,state: np.ndarray) -> np.ndarray:
        """Apply the transition to a state."""
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.edge_attr.device,
        )

        with torch.no_grad():
            prediction = self.forward(state_tensor)

        return prediction.detach().cpu().numpy()