"""Training logic for the GNN."""
import torch
import torch.nn as nn

def train_water_network(
    model,
    edge_index,
    edge_attr,
    edge_flows,
    node_concentrations_matrix,
    directed_link_quality,
    pipe_edge_mask,
    non_source_nodes_mask=[],
    epochs=100,
    unroll_steps=10,
    lr=0.001,
    link_loss_weight=1.0,
    buffer_spread_weight=0.01
):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    concentration_criterion = nn.MSELoss()  # Prediction Accuracy
    link_quality_criterion = nn.MSELoss()
    advection_criterion = nn.L1Loss()  # Conservation of Concentrations

    num_edges = edge_index.size(1)
    total_timesteps = node_concentrations_matrix.size(0)

    pipe_areas = torch.pi * (edge_attr[:, 1] / 2.0) ** 2
    pipe_lengths = edge_attr[:, 0]
    dt_simulation = 3600.0
    dx_buffer_segment = pipe_lengths / model.buffer_size

    for epoch in range(epochs):
        edge_buffer = torch.zeros(num_edges, model.buffer_size, device=node_concentrations_matrix.device)
        epoch_loss = 0.0
        epoch_node_loss = 0.0
        epoch_link_loss = 0.0
        epoch_spread_loss = 0.0

        for t in range(0, total_timesteps - 1, unroll_steps):
            loss = 0.0
            for step in range(t, min(t + unroll_steps,
                                     total_timesteps - 1)):  # Run model sequentially over the unroll window
                current_x = node_concentrations_matrix[step]
                target_x = node_concentrations_matrix[step + 1]

                current_buffer = edge_buffer.clone().detach()

                concentration_aggregate, next_edge_buffer, _ = model(current_x, edge_index, edge_buffer, torch.stack(
                    [edge_flows[step], edge_attr[:, 0], edge_attr[:, 1]], axis=-1).to(
                    node_concentrations_matrix.device))

                concentration_loss = concentration_criterion(concentration_aggregate * non_source_nodes_mask,
                                                             target_x * non_source_nodes_mask)  # Calculate loss against node ground truth, EXCLUDING source nodes

                predicted_link_quality = next_edge_buffer.mean(dim=1, keepdim=True)
                current_flows = edge_flows[step]
                active_pipe_mask = (current_flows > 0) & pipe_edge_mask

                if active_pipe_mask.any():
                    target_link_quality = directed_link_quality[step + 1]

                    link_quality_loss = (
                        link_quality_criterion(
                            predicted_link_quality[active_pipe_mask],
                            target_link_quality[active_pipe_mask]
                        )
                    )

                    active_buffer = next_edge_buffer[active_pipe_mask]
                    active_buffer_mean = active_buffer.mean(dim=1, keepdim=True)
                    buffer_spread_loss = ((active_buffer - active_buffer_mean).pow(2).mean())
                else:
                    link_quality_loss = torch.tensor(0.0, device=node_concentrations_matrix.device)
                    buffer_spread_loss = torch.tensor(0.0, device=node_concentrations_matrix.device)

                current_flows = edge_flows[step]
                velocities = current_flows / pipe_areas
                temporal_diff = (next_edge_buffer - current_buffer) / dt_simulation
                spatial_diff = (next_edge_buffer[:, 1:] - next_edge_buffer[:, :-1]) / dx_buffer_segment.unsqueeze(1)

                advection_loss = advection_criterion(temporal_diff[:, :-1] + velocities.unsqueeze(1) * spatial_diff,
                                                     torch.zeros_like(temporal_diff[:, :-1]))

                loss += (
                        concentration_loss
                        + link_loss_weight
                        * link_quality_loss
                        + buffer_spread_weight
                        * buffer_spread_loss
                ) # +  0.1 * advection_loss
                epoch_node_loss += concentration_loss.item()
                epoch_link_loss += link_quality_loss.item()
                epoch_spread_loss += buffer_spread_loss.item()
                edge_buffer = next_edge_buffer  # Pass the updated buffers forward to the next timestep

            optimizer.zero_grad()
            loss.backward()  # Backpropagate through the unrolled time window
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            edge_buffer = edge_buffer.detach()  # Detach buffer history so gradients don't leak into the next window block
            epoch_loss += loss.item()
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Total: {epoch_loss:.6f}, "
            f"Node: {epoch_node_loss:.6f}, "
            f"Link: {epoch_link_loss:.6f}, "
            f"Spread: {epoch_spread_loss:.6f}"
        )

def train_flow_reconstruction(model, edge_index, edge_attr, edge_flows, node_concentrations_matrix,
                              non_source_nodes_mask=[], epochs=100, unroll_steps=10, lr=0.001):
    model.eval()
    device = node_concentrations_matrix.device

    persistent_flows = edge_attr[:, 1].clone().detach().to(device)
    persistent_buffers = torch.zeros(edge_index.size(1), model.buffer_size, device=device)
    persistent_concentrations = node_concentrations_matrix[0].clone().detach().to(device)

    # Pre-compute static structural masks
    flow_sensor_mask = (edge_flows != 0).to(device)
    node_sensor_mask = (node_concentrations_matrix != 0).to(device)

    for epoch in range(epochs):
        input_flows = persistent_flows.clone().detach().requires_grad_(True)
        input_buffers = persistent_buffers.clone().detach().requires_grad_(True)
        concentration_aggregate_in = persistent_concentrations.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([input_flows, input_buffers, concentration_aggregate_in], lr=lr)

        flow_criterion = nn.MSELoss()
        concentration_criterion = nn.MSELoss()
        loss = 0.0

        for step in range(len(edge_flows) - 1):
            current_gt_flow = edge_flows[step].to(device)
            current_gt_conc = node_concentrations_matrix[step].to(device)

            active_flows = torch.where(flow_sensor_mask[step], current_gt_flow, input_flows)
            active_conc = torch.where(node_sensor_mask[step], current_gt_conc, concentration_aggregate_in)

            edge_attr_dynamic = torch.stack([
                active_flows,
                edge_attr[:, 0].to(device),
                edge_attr[:, 1].to(device)
            ], dim=-1)

            concentration_aggregate_in, input_buffers, predicted_flows = model(
                active_conc,
                edge_index.to(device),
                input_buffers,
                edge_attr_dynamic
            )

            next_gt_flow = edge_flows[step + 1].to(device)
            next_gt_conc = node_concentrations_matrix[step + 1].to(device)

            loss += flow_criterion(predicted_flows, next_gt_flow)
            loss += concentration_criterion(node_sensor_mask[step + 1] * concentration_aggregate_in,
                                            node_sensor_mask[step + 1] * next_gt_conc)

        optimizer.zero_grad()
        loss.backward()

        if input_flows.grad is not None:
            # Mask out global sensor locations across the window
            global_flow_mask = (edge_flows.sum(dim=0) != 0).to(device)
            input_flows.grad *= (~global_flow_mask).float()

        torch.nn.utils.clip_grad_norm_([input_flows, input_buffers, concentration_aggregate_in], max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            persistent_flows = input_flows.clone().detach()
            persistent_buffers = input_buffers.clone().detach()
            persistent_concentrations = concentration_aggregate_in.clone().detach()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}')
