"""Prepare EPyT-Flow data for the GNN."""

import torch
from epyt_flow.simulation import ScadaData


def prepare_gnn_data(scada_data: ScadaData):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    topo = scada_data.get_attributes()['network_topo']
    idx_to_node = {idx: name for idx, name in enumerate(topo.nodes)}

    edge_attr = []
    edge_index = torch.tensor(scada_data.get_topo_edge_indices(), dtype=torch.long).to(device)
    edge_flows = torch.tensor(
        scada_data.get_data_flows_as_edge_features()[0],
        dtype=torch.float,
    ).to(device)
    for edge in zip(edge_index[0], edge_index[1]):
        u, v = (edge[0], edge[1])
        e = topo.get_edge_data(idx_to_node[int(u)], idx_to_node[int(v)])
        edge_attr.append([e['info']['diameter'], e['length']])

    edge_attr = torch.tensor(edge_attr, dtype=torch.float).to(device)
    node_concentrations_matrix = torch.tensor(scada_data.node_quality_data_raw, dtype=torch.float32,).unsqueeze(-1).to(device)
    node_to_idx = {str(name): idx for idx, name in idx_to_node.items()}
    source_mask = [ node_to_idx[str(node_id)] for node_id in topo.get_all_reservoirs()]
    non_source_nodes_mask = torch.ones(node_concentrations_matrix.size(1), dtype=torch.bool)

    if source_mask:
        non_source_nodes_mask[source_mask] = False

    non_source_nodes_mask = non_source_nodes_mask.unsqueeze(-1).to(node_concentrations_matrix.device)

    return (
        device,
        edge_attr,
        edge_flows,
        edge_index,
        node_concentrations_matrix,
        non_source_nodes_mask,
    )

def prepare_gnn_link_quality_data(scada_data: ScadaData, edge_index: torch.Tensor, device: torch.device):
    topo = scada_data.get_attributes()["network_topo"]
    idx_to_node = {idx: name for idx, name in enumerate(topo.nodes)}
    physical_links = list(topo.get_all_links())
    link_id_to_column = {str(link_id): column for column, (link_id, _) in enumerate(physical_links)}
    link_quality = torch.tensor(scada_data.get_data_links_quality(), dtype=torch.float32, device=device)
    directed_link_columns = []
    pipe_edge_mask = []

    for edge in zip(edge_index[0], edge_index[1]):
        u_idx = int(edge[0])
        v_idx = int(edge[1])
        u = idx_to_node[u_idx]
        v = idx_to_node[v_idx]
        info = topo.get_edge_data(u, v)["info"]
        link_id = str(info["id"])
        directed_link_columns.append(link_id_to_column[link_id])
        pipe_edge_mask.append(info["type"] == 1)

    directed_link_columns = torch.tensor(directed_link_columns, dtype=torch.long, device=device)
    pipe_edge_mask = torch.tensor(pipe_edge_mask, dtype=torch.bool, device=device)
    directed_link_quality = link_quality[:, directed_link_columns].unsqueeze(-1)

    return directed_link_quality, pipe_edge_mask
