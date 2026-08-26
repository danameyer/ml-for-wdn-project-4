"""GNN training and checkpoint creation."""

from pathlib import Path
import torch
from epyt_flow.simulation import ScadaData
from wdn_kalman.datasets import Dataset
from wdn_kalman.gnn.data import prepare_gnn_data, prepare_gnn_link_quality_data
from wdn_kalman.gnn.model import GNN
from wdn_kalman.gnn.training import train_water_network
from wdn_kalman.paths import ProjectPaths


class GNNManager:
    """Train, save and load GNN models."""

    def __init__(self, paths: ProjectPaths):
        self.paths = paths

    def training_scada_file(self, dataset: Dataset) -> Path:
        return (
            self.paths.data_dir / f"{dataset.file_prefix}_randDemand=True_training.epytflow_scada_data"
        )

    def gnn_file(self, dataset: Dataset, buffer_size: int, unroll_steps: int) -> Path:
        return (
            self.paths.models_dir / f"{dataset.file_prefix}_gnn_buffer_size={buffer_size}_unroll_steps={unroll_steps}.pt"
        )

    def train(
        self,
        dataset: Dataset,
        buffer_size: int = 5,
        unroll_steps: int = 20,
        epochs: int = 1000,
        lr: float = 0.001,
        link_loss_weight: float = 1.0,
        buffer_spread_weight: float = 0.01,
        overwrite: bool = False,
    ) -> Path:
        model_file = self.gnn_file(dataset=dataset, buffer_size=buffer_size, unroll_steps=unroll_steps)

        if model_file.exists() and not overwrite:
            return model_file

        scada_data = ScadaData.load_from_file(str(self.training_scada_file(dataset)))

        (
            device,
            edge_attr,
            edge_flows,
            edge_index,
            node_concentrations_matrix,
            non_source_nodes_mask,
        ) = prepare_gnn_data(scada_data)

        (
            directed_link_quality,
            pipe_edge_mask,
        ) = prepare_gnn_link_quality_data(scada_data=scada_data, edge_index=edge_index, device=device)

        model = GNN(buffer_size=buffer_size).to(device)

        print('Starting training pipeline...')
        train_water_network(
            model=model,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_flows=edge_flows,
            node_concentrations_matrix=node_concentrations_matrix,
            directed_link_quality=directed_link_quality,
            pipe_edge_mask=pipe_edge_mask,
            non_source_nodes_mask=non_source_nodes_mask,
            epochs=epochs,
            unroll_steps=unroll_steps,
            lr=lr,
            link_loss_weight=link_loss_weight,
            buffer_spread_weight=buffer_spread_weight
        )
        print('Training completed successfully.')

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "buffer_size": buffer_size,
            "unroll_steps": unroll_steps,
            "edge_index": edge_index.detach().cpu(),
            "edge_attr": edge_attr.detach().cpu(),
            "link_loss_weight": link_loss_weight
        }

        model_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, model_file)

        return model_file

    def load(
        self,
        dataset: Dataset,
        buffer_size: int,
        unroll_steps: int,
        device: torch.device,
    ) -> tuple[GNN, dict]:
        model_file = self.gnn_file(dataset=dataset, buffer_size=buffer_size, unroll_steps=unroll_steps)
        checkpoint = torch.load(model_file, map_location=device, weights_only=False)
        model = GNN(buffer_size=checkpoint["buffer_size"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return model, checkpoint
