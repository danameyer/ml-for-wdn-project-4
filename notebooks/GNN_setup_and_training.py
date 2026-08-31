import marimo

__generated_with = "0.19.4"
app = marimo.App(width="full")


@app.cell
def _():
    from epyt_flow.data.benchmarks import load_leakdb_scenarios
    from epyt_flow.simulation import ScenarioSimulator, flowunit_to_str
    from epyt_flow.utils import to_seconds
    from epyt_flow.visualization import ScenarioVisualizer

    import os
    from pathlib import Path
    import numpy as np
    from epyt_flow.simulation import EpanetConstants, ModelUncertainty, \
        ScenarioConfig, ScadaData, SensorConfig
    from epyt_flow.uncertainty import RelativeUniformUncertainty, AbsoluteGaussianUncertainty
    from epyt_control.envs import HydraulicControlEnv
    from epyt_control.envs.actions import ChemicalInjectionAction
    return (
        AbsoluteGaussianUncertainty,
        ChemicalInjectionAction,
        EpanetConstants,
        HydraulicControlEnv,
        ModelUncertainty,
        Path,
        RelativeUniformUncertainty,
        ScadaData,
        ScenarioConfig,
        ScenarioSimulator,
        SensorConfig,
        load_leakdb_scenarios,
        np,
        os,
        to_seconds,
    )


@app.cell
def _():
    import networkx as nx
    import matplotlib.pyplot as plt
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    from torch_geometric.data import Data
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import from_networkx, scatter
    return MessagePassing, nn, scatter, torch


@app.cell
def _(
    AbsoluteGaussianUncertainty,
    ChemicalInjectionAction,
    EpanetConstants,
    HydraulicControlEnv,
    ModelUncertainty,
    Path,
    RelativeUniformUncertainty,
    ScadaData,
    ScenarioConfig,
    ScenarioSimulator,
    SensorConfig,
    load_leakdb_scenarios,
    np,
    os,
    to_seconds,
):
    """
    Create data sets.
    """
    path_to_scenarios = 'data'

    def create_leakdb_scenario(use_net1: bool=False, randomized_demands: bool=False) -> None:
        [scenario_config] = load_leakdb_scenarios(scenarios_id=list(range(1)), use_net1=use_net1)
        with ScenarioSimulator(scenario_config=scenario_config) as sim:
            sim.set_general_parameters(simulation_duration=to_seconds(days=120))
            if randomized_demands is True:
                sim.randomize_demands()
            sim.enable_chemical_analysis()
            reservoid_node_id = sim.epanet_api.get_all_reservoirs_id()[0]
            sim.add_quality_source(node_id=reservoid_node_id, pattern=np.array([1.0]), source_type=EpanetConstants.EN_CONCEN, pattern_id='my-chl-injection')
            for node_idx in sim.epanet_api.get_all_nodes_idx():
                sim.epanet_api.set_node_init_quality(node_idx, 0)
            for link_idx in sim.epanet_api.get_all_links_idx():
                sim.epanet_api.setlinkvalue(link_idx, EpanetConstants.EN_KBULK, -0.5)  # Create scenarios based on the LeakDB Hanoi
                sim.epanet_api.setlinkvalue(link_idx, EpanetConstants.EN_KWALL, -0.01)
            sim.sensor_config = SensorConfig.create_empty_sensor_config(sim.sensor_config)
            sim.set_pressure_sensors(sim.sensor_config.nodes)
            sim.set_demand_sensors(sim.sensor_config.nodes)
            sim.set_flow_sensors(sim.sensor_config.links)
            sim.set_node_quality_sensors(sim.sensor_config.nodes)
            sim.set_link_quality_sensors(sim.sensor_config.links)
            my_uncertainties = {'global_pipe_length_uncertainty': RelativeUniformUncertainty(low=0.8, high=1.8), 'global_pipe_roughness_uncertainty': RelativeUniformUncertainty(low=0.8, high=1.8), 'global_base_demand_uncertainty': RelativeUniformUncertainty(low=0.8, high=1.8), 'global_demand_pattern_uncertainty': AbsoluteGaussianUncertainty(mean=0, scale=0.02)}  # Enable chlorine simulation and place a chlorine injection pump at the reservoir
            sim.set_model_uncertainty(ModelUncertainty(**my_uncertainties))
            Path(path_to_scenarios).mkdir(exist_ok=True)
            sim.save_to_epanet_file(os.path.join(path_to_scenarios, f'control_cl_injection_scenario-Net1={use_net1}_randDemand={randomized_demands}.inp'))
            sim.get_scenario_config().save_to_file(os.path.join(path_to_scenarios, f'control_cl_injection_scenario-Net1={use_net1}_randDemand={randomized_demands}'))

    class LeakdDbChlorineInjectionEnv(HydraulicControlEnv):

        def __init__(self, use_net1: bool=False, randomized_demands: bool=False):
            scenario_config_file_in = os.path.join(path_to_scenarios, f'control_cl_injection_scenario-Net1={use_net1}_randDemand={randomized_demands}.epytflow_scenario_config')  # Set initial concentration and simple (constant) reactions
            injection_node_id = '1'
            if use_net1 is True:
                injection_node_id = '9'
            super().__init__(scenario_config=ScenarioConfig.load_from_file(scenario_config_file_in), chemical_injection_actions=[ChemicalInjectionAction(node_id=injection_node_id, pattern_id='my-chl-injection', source_type_id=EpanetConstants.EN_CONCEN, upper_bound=5.0)], autoreset=False, reload_scenario_when_reset=False)

        def _compute_reward_function(self, scada_data: ScadaData) -> float:
            return 0  # Set flow and chlorine sensors everywhere

    def create_data_set(use_net1: bool, randomized_demands: bool, file_out: str, path_out: str='data') -> None:
        scada_data = None
        control_actions = []
        with LeakdDbChlorineInjectionEnv(use_net1, randomized_demands) as env:
            env.reset()
            for _ in range(1000):
                action = env.action_space.sample()  # Specify uncertainties -- similar to the one already implemented in LeakDB
                control_actions.append(action)
                _, _, terminated, _, info = env.step(action)
                if terminated is True:
                    break
                current_scada_data = info['scada_data']
                if scada_data is None:
                    scada_data = current_scada_data  # Export scenario
                else:
                    scada_data.concatenate(current_scada_data)
            env.close()
        Path(path_out).mkdir(exist_ok=True)
        scada_data.save_to_file(os.path.join(path_out, f'{file_out}.epytflow_scada_data'))
        np.savez(os.path.join(path_out, f'{file_out}.npz'), control_actions=control_actions)
        return scada_data
    return (create_data_set,)


@app.cell
def _(create_data_set):
    #Hanoi
    #create_leakdb_scenario(use_net1=False, randomized_demands=False)  # Load scenario and set autoreset=True
    hftr = create_data_set(use_net1=False, randomized_demands=False, file_out='hanoi_randDemand=False_training')
    hfva = create_data_set(use_net1=False, randomized_demands=False, file_out='hanoi_randDemand=False_validation')
    hfte = create_data_set(use_net1=False, randomized_demands=False, file_out='hanoi_randDemand=False_test')
    #create_leakdb_scenario(use_net1=False, randomized_demands=True)
    httr = create_data_set(use_net1=False, randomized_demands=True, file_out='hanoi_randDemand=True_training')
    htva = create_data_set(use_net1=False, randomized_demands=True, file_out='hanoi_randDemand=True_validation')
    htte = create_data_set(use_net1=False, randomized_demands=True, file_out='hanoi_randDemand=True_test')

    #Net1
    #create_leakdb_scenario(use_net1=True, randomized_demands=False)
    create_data_set(use_net1=True, randomized_demands=False, file_out='net1_randDemand=False_training')
    create_data_set(use_net1=True, randomized_demands=False, file_out='net1_randDemand=False_validation')
    create_data_set(use_net1=True, randomized_demands=False, file_out='net1_randDemand=False_test')
    #create_leakdb_scenario(use_net1=True, randomized_demands=True)
    create_data_set(use_net1=True, randomized_demands=True, file_out='net1_randDemand=True_training')
    create_data_set(use_net1=True, randomized_demands=True, file_out='net1_randDemand=True_validation')
    create_data_set(use_net1=True, randomized_demands=True, file_out='net1_randDemand=True_test')
    return hfte, hftr


@app.cell
def _(MessagePassing, nn, scatter, torch):
    class GNN(MessagePassing):
        def __init__(self, buffer_size = 5):
            super().__init__(aggr="add", flow ="source_to_target")
            self.buffer_size = buffer_size

            self.MLP = nn.Sequential(
                # source concentration + flow + length + width + buffer + incoming edge attributes + aggregrated buffer
                nn.Linear(1 + 3 + buffer_size + 3 + buffer_size, 32),  
                nn.Sigmoid(),
                nn.Linear(32, 32),
                nn.Sigmoid(),
                nn.Linear(32, buffer_size + 1) # buffer size + emitted concentration for the next time step
            )

        def forward(self, x, edge_index, edge_buffer, edge_attr):
            j, i = edge_index
            x_j = x[j] # Features of source nodes

            flow_directions = torch.sign(edge_attr[:, 0])

            pipe_lenghts = edge_attr[:, 1]
            pipe_diameters = edge_attr[:, 2]
            pipe_resistances = (10.67  * pipe_lenghts) / (pipe_diameters ** 4.87 + 1e-6)
            pipe_capacities = (1.0 / torch.clamp(pipe_resistances, min = 1e-6)) ** 0.54

            outgoing_capacities = scatter(pipe_capacities, j, dim_size=x.size(0), reduce='sum')
            fraction_capacities = pipe_capacities / (outgoing_capacities[j] + 1e-6)
        
            incoming_flows = scatter(nn.functional.relu(edge_attr[:, 0]), i, dim=0, dim_size=x.size(0), reduce='sum').reshape(-1, 1)
            outgoing_flows = incoming_flows[j].reshape(-1) * fraction_capacities * flow_directions

            incoming_buffers = scatter(edge_buffer, i, dim=0, dim_size=x.size(0), reduce='add')
            incoming_attrs = scatter(edge_attr[:, 1:], i, dim=0, dim_size=x.size(0), reduce='mean')

            input = torch.cat([x_j, edge_attr, edge_buffer, incoming_flows[j], incoming_attrs[j], incoming_buffers[j]], dim = -1)
            output = self.MLP(input)

            outgoing_buffers = output[:, :self.buffer_size]
            incoming_concentrations = output[:, -1:]

            outgoing_concentrations = self.propagate(edge_index, edge_msg=incoming_concentrations, size=(x.size(0), x.size(0)))
            return outgoing_concentrations, outgoing_buffers, outgoing_flows

        def message(self, edge_msg):
            return edge_msg
    return (GNN,)


@app.cell
def _(nn, torch):
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

        min_loss = 99999999
        for epoch in range(epochs):
            edge_buffer = torch.zeros(num_edges, model.buffer_size, device=node_concentrations_matrix.device)
            epoch_loss = 0.0
            epoch_node_loss = 0.0
            epoch_link_loss = 0.0
            epoch_spread_loss = 0.0

            for t in range(0, total_timesteps - 1, unroll_steps):
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

    return (train_water_network,)


@app.cell
def _(nn, torch):
    def train_flow_reconstruction(model, edge_index, edge_attr, edge_flows, node_concentrations_matrix, non_source_nodes_mask=[], flow_sensor_mask = [], node_sensor_mask = [], epochs=100, unroll_steps=10, lr=0.001):
        model.eval()
    
        device = node_concentrations_matrix.device
        flow_criterion = nn.MSELoss()
        concentration_criterion = nn.MSELoss()

        flow_sensor_mask_inv = ~flow_sensor_mask
        node_sensor_mask_inv = ~node_sensor_mask

        input_flows = (torch.rand((sum(flow_sensor_mask_inv)), generator=torch.manual_seed(123456789)) * max(edge_flows[0])).to(device).requires_grad_(True)
        input_concentrations = torch.rand((sum(node_sensor_mask_inv), 1), generator=torch.manual_seed(123456789)).clone().detach().to(device).requires_grad_(True)
        input_buffers = torch.rand(edge_index.size(1), model.buffer_size, generator=torch.manual_seed(123456789),  device=device).detach().requires_grad_(True)
    
        optimizer = torch.optim.Adam([input_flows, input_buffers, input_concentrations], lr=lr)

        for epoch in range(epochs):
            loss = 0.0
            predicted_flows = edge_flows[0].clone().detach().to(device).requires_grad_(False)
            predicted_flows[~flow_sensor_mask] =  input_flows
        
            predicted_concentrations = node_concentrations_matrix[0].clone().detach().to(device).requires_grad_(False)
            predicted_concentrations[~node_sensor_mask.flatten()] = input_concentrations

        
            predicted_buffers = input_buffers
        
            for step in range(20):

                predicted_flows = torch.where(flow_sensor_mask, edge_flows[step], predicted_flows)
                predicted_concentrations = torch.where(node_sensor_mask, node_concentrations_matrix[step], predicted_concentrations)
            
                edge_attr_dynamic = torch.stack([
                    predicted_flows, 
                    edge_attr[:, 0].to(device), 
                    edge_attr[:, 1].to(device)
                ], dim=-1)

                predicted_concentrations, predicted_buffers, predicted_flows = model(
                    predicted_concentrations, 
                    edge_index.to(device), 
                    predicted_buffers, 
                    edge_attr_dynamic
                )

                next_gt_flow = edge_flows[step + 1].to(device)
                next_gt_conc = node_concentrations_matrix[step + 1].to(device)


                loss += flow_criterion(predicted_flows * flow_sensor_mask_inv, next_gt_flow * flow_sensor_mask_inv)
                loss += concentration_criterion(predicted_concentrations, next_gt_conc)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_([input_flows, input_buffers, input_concentrations], max_norm=1.0)
            optimizer.step()


            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}')
    return (train_flow_reconstruction,)


@app.cell
def _(hfte, hftr, torch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    topo = hfte.get_attributes()['network_topo']
    idx_to_node = {idx: name for idx, name in enumerate(topo.nodes)}
    node_to_idx = {name: idx for idx, name in enumerate(topo.nodes)}
    physical_links = list(topo.get_all_links())
    
    link_id_to_column = {str(link_id): column for column, (link_id, _) in enumerate(physical_links)}
    

    edge_attr = []
    flow_sensor_mask = []

    edge_index = torch.tensor(hfte.get_topo_edge_indices(), dtype=torch.long).to(device)
    edge_flows = torch.tensor(hfte.get_data_flows_as_edge_features()[0], dtype=torch.float).to(device)
    link_quality = torch.tensor(hfte.get_data_links_quality(), dtype=torch.float32, device=device)
    directed_link_columns = []
    

    pipe_edge_mask = []

    for edge in zip(edge_index[0], edge_index[1]):
        u, v = (edge[0], edge[1])
        e = topo.get_edge_data(idx_to_node[int(u)], idx_to_node[int(v)])
        e_id = e['info']['id']
    
        edge_attr.append([e['info']['diameter'], e['length']])
        flow_sensor_mask.append(e_id in hfte.sensor_config.links)
    

        directed_link_columns.append(link_id_to_column[e['info']['id']])
        pipe_edge_mask.append(e['info']['type'] == 1)

    edge_attr = torch.tensor(edge_attr, dtype=torch.float).to(device)

    flow_sensor_mask = torch.tensor(flow_sensor_mask).flatten()
    node_sensor_mask = torch.zeros(topo.number_of_nodes(), dtype=torch.bool).flatten()
    node_sensor_mask[[node_to_idx[i] for i in hftr.sensor_config.quality_node_sensors]] = True
    node_sensor_mask = node_sensor_mask.reshape((-1, 1))

    node_concentrations_matrix = torch.tensor(hftr.node_quality_data_raw, dtype=torch.float32).unsqueeze(-1).to(device)

    source_mask = [node_to_idx[x] for x in hftr.get_attributes()['network_topo'].get_all_reservoirs()]
    non_source_nodes_mask = torch.ones(node_concentrations_matrix.size(1), dtype=torch.bool)
    if source_mask:
        non_source_nodes_mask[source_mask] = False
    non_source_nodes_mask = non_source_nodes_mask.unsqueeze(-1).to(node_concentrations_matrix.device)

    directed_link_columns = torch.tensor(directed_link_columns, dtype=torch.long, device=device)
    pipe_edge_mask = torch.tensor(pipe_edge_mask, dtype=torch.bool, device=device)
    directed_link_quality = link_quality[:, directed_link_columns].unsqueeze(-1)

    return (
        device,
        directed_link_quality,
        edge_attr,
        edge_flows,
        edge_index,
        flow_sensor_mask,
        node_concentrations_matrix,
        node_sensor_mask,
        non_source_nodes_mask,
        pipe_edge_mask,
    )


@app.cell
def _():
    NUM_NODES = 32
    NUM_EDGES = 68
    TOTAL_TIMESTEPS = 1000 # Total length of your simulation time-series
    BUFFER_SIZE = 5  # Number of discrete slots inside each edge buffer
    UNROLL_STEPS = 20  # How many timesteps to track before updating gradients
    EPOCHS = 1000  
    return


@app.cell
def _(
    GNN,
    buffer,
    device,
    directed_link_quality,
    edge_attr,
    edge_flows,
    edge_index,
    node_concentrations_matrix,
    non_source_nodes_mask,
    pipe_edge_mask,
    results,
    rollout,
    train_water_network,
):
    model = GNN(buffer_size=25).to(device)
    print('Starting training pipeline...')

    results[buffer][rollout] = train_water_network(model=model, 
                edge_index=edge_index, 
                edge_attr=edge_attr, 
                edge_flows=edge_flows, 
                node_concentrations_matrix=node_concentrations_matrix, 
                non_source_nodes_mask=non_source_nodes_mask,
                directed_link_quality=directed_link_quality,
                pipe_edge_mask=pipe_edge_mask,
                epochs=500, 
                unroll_steps=25, 
                lr=0.05)
    print('Training completed successfully!')
    return (model,)


@app.cell
def _(
    edge_attr,
    edge_flows,
    edge_index,
    flow_sensor_mask,
    i,
    model,
    node_concentrations_matrix,
    node_sensor_mask,
    non_source_nodes_mask,
    train_flow_reconstruction,
):
    train_flow_reconstruction(model=model, 
                            edge_index=edge_index, 
                            edge_attr=edge_attr, 
                            edge_flows=edge_flows, 
                            node_concentrations_matrix=node_concentrations_matrix, 
                            non_source_nodes_mask=non_source_nodes_mask,
                            node_sensor_mask=node_sensor_mask,
                            flow_sensor_mask=flow_sensor_mask,                           
                            epochs=2000, 
                            unroll_steps=i, 
                            lr=5)
    return


if __name__ == "__main__":
    app.run()
