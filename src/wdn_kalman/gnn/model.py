"""Graph neural network model."""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

class GNN(MessagePassing):
    def __init__(self, buffer_size=5):
        super().__init__(aggr="add", flow="source_to_target")
        self.buffer_size = buffer_size

        self.MLP = nn.Sequential(
            # source concentration + flow + length + width + buffer + incoming edge attributes + aggregrated buffer
            nn.Linear(1 + 3 + buffer_size + 3 + buffer_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, buffer_size + 1)  # buffer size + emitted concentration for the next time step
        )

    def forward(self, x, edge_index, edge_buffer, edge_attr):
        j, i = edge_index
        x_j = x[j]  # Features of source nodes

        flow_directions = torch.sign(edge_attr[:, 0])

        incoming_buffers = scatter(edge_buffer, i, dim=0, dim_size=x.size(0), reduce='add')
        incoming_flows = scatter(nn.functional.relu(edge_attr[:, 0]), i, dim=0, dim_size=x.size(0),
                                 reduce='sum').reshape(-1, 1)
        incoming_attrs = scatter(edge_attr[:, 1:], i, dim=0, dim_size=x.size(0), reduce='mean')
        outgoing_attrs = scatter(edge_attr, j, dim=0, dim_size=x.size(0), reduce='add')

        input = torch.cat([x_j, edge_attr, edge_buffer, incoming_flows[j], incoming_attrs[j], incoming_buffers[j]],
                          dim=-1)
        output = self.MLP(input)

        next_buffer = output[:, :self.buffer_size]
        concentration = output[:, -1:]

        # Propagate the edge-level concentration messages to the target nodes
        concentration_aggregate = self.propagate(edge_index, edge_msg=concentration, size=(x.size(0), x.size(0)))
        predicted_flow = incoming_flows[j] * (edge_attr[j, 1] / incoming_attrs[j, 1]) * flow_directions[j]

        return concentration_aggregate, next_buffer, predicted_flow

    def message(self, edge_msg):
        return edge_msg