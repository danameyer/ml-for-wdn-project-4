"""Evaluation of the trained GNN without Kalman filter."""

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from epyt_flow.simulation import ScadaData
from wdn_kalman.datasets import Dataset
from wdn_kalman.gnn.gnn_state_estimation import load_gnn_state_estimation_data
from wdn_kalman.gnn.manager import GNNManager
from wdn_kalman.paths import ProjectPaths


@dataclass
class GNNEvaluationResult:
    """GNN chlorine prediction results."""
    dataset: Dataset
    chlorine_predictions: np.ndarray
    chlorine_true: np.ndarray
    mae_per_location: np.ndarray
    node_ids: tuple[str, ...]
    link_ids: tuple[str, ...]
    buffer_size: int
    unroll_steps: int
    result_file: Path

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_links(self) -> int:
        return len(self.link_ids)

    @property
    def mean_mae(self) -> float:
        return float(np.mean(self.mae_per_location))

    @property
    def std_mae(self) -> float:
        return float(np.std(self.mae_per_location))

    @property
    def node_mae_per_location(self) -> np.ndarray:
        return self.mae_per_location[:self.num_nodes]

    @property
    def link_mae_per_location(self) -> np.ndarray:
        return self.mae_per_location[self.num_nodes:]

    @property
    def node_mean_mae(self) -> float:
        return float(np.mean(self.node_mae_per_location))

    @property
    def node_std_mae(self) -> float:
        return float(np.std(self.node_mae_per_location))

    @property
    def link_mean_mae(self) -> float:
        return float(np.mean(self.link_mae_per_location))

    @property
    def link_std_mae(self) -> float:
        return float(np.std(self.link_mae_per_location))


def calculate_mae_per_location(chlorine_predictions: np.ndarray, true_chlorine_values: np.ndarray) -> np.ndarray:
    """Calculate MAE over time for every location."""
    chlorine_predictions = np.asarray(chlorine_predictions, dtype=float)
    true_chlorine_values = np.asarray(true_chlorine_values, dtype=float)

    return np.mean(np.abs(chlorine_predictions - true_chlorine_values), axis=0)


def gnn_evaluation_file(paths: ProjectPaths, dataset: Dataset, buffer_size: int, unroll_steps: int) -> Path:
    """Return output file for GNN evaluation."""
    return paths.gnn_evaluation_dir / f"{dataset.file_prefix}_gnn_buffer_size={buffer_size}_unroll_steps={unroll_steps}_evaluation.npz"


def save_gnn_evaluation_result(result: GNNEvaluationResult) -> Path:
    """Save GNN predictions and MAE metrics."""
    result.result_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        result.result_file,
        dataset=np.asarray(result.dataset.net_name),
        chlorine_predictions=np.asarray(result.chlorine_predictions, dtype=float),
        chlorine_true=np.asarray(result.chlorine_true, dtype=float),
        mae_per_location=np.asarray(result.mae_per_location, dtype=float),
        node_ids=np.asarray(result.node_ids, dtype=str),
        link_ids=np.asarray(result.link_ids, dtype=str),
        buffer_size=np.asarray(result.buffer_size, dtype=int),
        unroll_steps=np.asarray(result.unroll_steps, dtype=int)
    )

    metrics_file = result.result_file.with_suffix(".json")

    metrics = {
        "dataset": result.dataset.net_name,
        "buffer_size": result.buffer_size,
        "unroll_steps": result.unroll_steps,
        "mean_mae": result.mean_mae,
        "std_mae": result.std_mae,
        "node_mean_mae": result.node_mean_mae,
        "node_std_mae": result.node_std_mae,
        "link_mean_mae": result.link_mean_mae,
        "link_std_mae": result.link_std_mae,
        "node_mae_per_location": {
            node_id: float(mae) for node_id, mae in zip(result.node_ids, result.node_mae_per_location)
        },
        "link_mae_per_location": {
            link_id: float(mae) for link_id, mae in zip(result.link_ids, result.link_mae_per_location)
        }
    }

    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return result.result_file


class GNNEvaluator:
    """Evaluate trained GNN without a Kalman filter."""

    def __init__(self, gnn_manager: GNNManager):
        self.gnn_manager = gnn_manager

    def run(self, dataset: Dataset, buffer_size: int, unroll_steps: int) -> GNNEvaluationResult:
        """Evaluate node and link chlorine predictions."""
        data = (
            load_gnn_state_estimation_data(
                gnn_manager=self.gnn_manager,
                dataset=dataset,
                buffer_size=buffer_size,
                unroll_steps=unroll_steps,
            )
        )

        state = data.create_initial_state()
        node_predictions = []
        link_predictions = []
        node_true = []
        link_true = []
        num_steps = len(data.node_concentrations) - 1

        for target_index in range(1, num_steps + 1):
            input_index = target_index - 1
            state[data.wrapper.node_slice] = data.node_concentrations[input_index]
            state[data.wrapper.flow_slice] = data.edge_flows[input_index]
            next_state = data.wrapper.predict_with_numpy_array(state)
            predicted_nodes = next_state[data.wrapper.node_slice]
            predicted_links = data.link_concentration_estimates(state=next_state, time_index=input_index)
            node_predictions.append(predicted_nodes)
            link_predictions.append(predicted_links)
            node_true.append(data.node_concentrations[target_index])
            link_true.append(data.link_concentrations[target_index])
            state = next_state

        node_predictions = np.vstack(node_predictions)
        link_predictions = np.vstack(link_predictions)
        node_true = np.vstack(node_true)
        link_true = np.vstack(link_true)
        chlorine_predictions = np.concatenate((node_predictions, link_predictions), axis=1)
        chlorine_true = np.concatenate((node_true, link_true), axis=1)
        mae_per_location = (
            calculate_mae_per_location(
                chlorine_predictions=chlorine_predictions,
                true_chlorine_values=chlorine_true,
            )
        )

        test_scada_file = self.gnn_manager.paths.data_dir / f"{dataset.file_prefix}_randDemand=False_test.epytflow_scada_data"
        scada = ScadaData.load_from_file(str(test_scada_file))
        node_ids = tuple(str(node_id) for node_id in scada.network_topo.get_all_nodes())
        link_ids = tuple(str(link_id) for link_id, _ in scada.network_topo.get_all_links())

        return GNNEvaluationResult(
            dataset=dataset,
            chlorine_predictions=chlorine_predictions,
            chlorine_true=chlorine_true,
            mae_per_location=mae_per_location,
            node_ids=node_ids,
            link_ids=link_ids,
            buffer_size=buffer_size,
            unroll_steps=unroll_steps,
            result_file=gnn_evaluation_file(
                paths=self.gnn_manager.paths,
                dataset=dataset,
                buffer_size=buffer_size,
                unroll_steps=unroll_steps,
            ),
        )