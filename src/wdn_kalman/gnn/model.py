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
            nn.Sigmoid(),
            nn.Linear(32, 32),
            nn.Sigmoid(),
            nn.Linear(32, buffer_size + 1)  # buffer size + emitted concentration for the next time step
        )

    def forward(self, x, edge_index, edge_buffer, edge_attr):
        j, i = edge_index
        x_j = x[j]  # Features of source nodes

        flow_directions = torch.sign(edge_attr[:, 0])

        pipe_diameters = edge_attr[:, 1]
        pipe_lengths = edge_attr[:, 2]
        pipe_resistances = (10.67 * pipe_lengths) / (pipe_diameters ** 4.87 + 1e-6)
        pipe_capacities = (1.0 / torch.clamp(pipe_resistances, min=1e-6)) ** 0.54

        outgoing_capacities = scatter(pipe_capacities, j, dim_size=x.size(0), reduce='sum')
        fraction_capacities = pipe_capacities / (outgoing_capacities[j] + 1e-6)

        incoming_flows = scatter(nn.functional.relu(edge_attr[:, 0]), i, dim=0, dim_size=x.size(0),
                                 reduce='sum').reshape(-1, 1)
        outgoing_flows = incoming_flows[j].reshape(-1) * fraction_capacities * flow_directions

        incoming_buffers = scatter(edge_buffer, i, dim=0, dim_size=x.size(0), reduce='add')
        incoming_attrs = scatter(edge_attr[:, 1:], i, dim=0, dim_size=x.size(0), reduce='mean')

        input = torch.cat([x_j, edge_attr, edge_buffer, incoming_flows[j], incoming_attrs[j], incoming_buffers[j]],
                          dim=-1)
        output = self.MLP(input)

        outgoing_buffers = output[:, :self.buffer_size]
        incoming_concentrations = output[:, -1:]

        outgoing_concentrations = self.propagate(edge_index, edge_msg=incoming_concentrations,
                                                 size=(x.size(0), x.size(0)))
        return outgoing_concentrations, outgoing_buffers, outgoing_flows

    def message(self, edge_msg):
        return edge_msg