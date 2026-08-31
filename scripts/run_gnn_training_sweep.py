"""Run GNN buffer-size and rollout-length sweep configuration."""

import argparse
import json
import random
import numpy as np
import torch
from epyt_flow.simulation import ScadaData
from wdn_kalman.datasets import DATASETS, get_dataset
from wdn_kalman.gnn.data import prepare_gnn_data, prepare_gnn_link_quality_data
from wdn_kalman.gnn.manager import GNNManager
from wdn_kalman.gnn.model import GNN
from wdn_kalman.gnn.training import train_water_network
from wdn_kalman.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one GNN training sweep configuration."
    )

    parser.add_argument(
        "--dataset",
        choices=DATASETS.keys(),
        required=True,
    )

    parser.add_argument(
        "--buffer-size",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--unroll-steps",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--link-loss-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--buffer-spread-weight",
        type=float,
        default=0.01,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    paths = ProjectPaths.from_project_root()
    dataset = get_dataset(args.dataset)
    manager = GNNManager(paths)
    scada_data = ScadaData.load_from_file(str(manager.training_scada_file(dataset)))

    (
        device,
        edge_attr,
        edge_flows,
        edge_index,
        node_concentrations_matrix,
        non_source_nodes_mask,
    ) = prepare_gnn_data(scada_data)

    (directed_link_quality, pipe_edge_mask) = prepare_gnn_link_quality_data(scada_data=scada_data, edge_index=edge_index, device=device)

    print(f"Running GNN sweep: dataset={dataset.net_name}, buffer={args.buffer_size}, rollout={args.unroll_steps}")

    model = GNN(buffer_size=args.buffer_size).to(device)
    losses = train_water_network(
        model=model,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_flows=edge_flows,
        node_concentrations_matrix=node_concentrations_matrix,
        directed_link_quality=directed_link_quality,
        pipe_edge_mask=pipe_edge_mask,
        non_source_nodes_mask=non_source_nodes_mask,
        epochs=args.epochs,
        unroll_steps=args.unroll_steps,
        lr=args.lr,
        link_loss_weight=args.link_loss_weight,
        buffer_spread_weight=args.buffer_spread_weight,
    )

    output_dir = paths.aggregated_results_dir / "gnn_training_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{dataset.file_prefix}_buffer={args.buffer_size}_rollout={args.unroll_steps}_seed={args.seed}.json"

    result = {
        "dataset": dataset.net_name,
        "buffer_size": args.buffer_size,
        "unroll_steps": args.unroll_steps,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "link_loss_weight": args.link_loss_weight,
        "buffer_spread_weight": args.buffer_spread_weight,
        **losses,
    }

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(f"Saved sweep result to: {output_file}")
    print(f"Table value: {losses['node_loss']:.6f}")


if __name__ == "__main__":
    main()