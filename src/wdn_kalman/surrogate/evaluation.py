"""Evaluation of the neural surrogate model."""
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from epyt_flow.simulation import ScadaData
from wdn_kalman.baseline_repository import load_baseline_module
from wdn_kalman.datasets import Dataset
from wdn_kalman.paths import ProjectPaths
from wdn_kalman.surrogate.manager import SurrogateManager


@dataclass
class SurrogateEvaluationResult:
    """Single step chlorine prediction results for a neural trained neural surrogate model."""
    dataset: Dataset
    chlorine_predictions: np.ndarray
    chlorine_true: np.ndarray
    mae_per_location: np.ndarray
    node_ids: tuple[str, ...]
    link_ids: tuple[str, ...]
    result_file: Path

    @property
    def num_nodes(self) -> int:
        """get number of nodes."""
        return len(self.node_ids)

    @property
    def num_links(self) -> int:
        """get number of links."""
        return len(self.link_ids)

    @property
    def mean_mae(self) -> float:
        """Mean of the time-averaged MAE over nodes and links."""
        return float(np.mean(self.mae_per_location))

    @property
    def std_mae(self) -> float:
        """standard deviation of time-averaged MAE over nodes and links."""
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

    def node_index(self, node_id: str) -> int:
        """Return the chlorine-column index for a node ID."""
        node_id = str(node_id)
        return self.node_ids.index(node_id)

def calculate_mae_per_location(chlorine_predictions: np.ndarray,
                               true_chlorine_values: np.ndarray
                               ) -> np.ndarray:
    """Calculate MAE over time for every location."""
    chlorine_predictions = np.asarray(chlorine_predictions, dtype=float)
    true_chlorine_values = np.asarray(true_chlorine_values, dtype=float)

    return np.mean(np.abs(chlorine_predictions - true_chlorine_values), axis=0)


def surrogate_evaluation_file(paths: ProjectPaths, dataset: Dataset) -> Path:
    """Return output file for neural surrogate model evaluation."""
    return paths.neural_surrogate_evaluation_dir / f"{dataset.file_prefix}_surrogate_evaluation.npz"

def surrogate_metrics_file(paths: ProjectPaths, dataset: Dataset) -> Path:
    return paths.neural_surrogate_evaluation_dir / f"{dataset.file_prefix}_surrogate_metrics.json"

def save_surrogate_evaluation_result(result: SurrogateEvaluationResult) -> Path:
    """Save surrogate predictions and metrics."""
    result.result_file.parent.mkdir(parents=True, exist_ok=True,)

    np.savez(
        result.result_file,
        dataset=np.asarray(result.dataset.net_name),
        chlorine_predictions=np.asarray(result.chlorine_predictions, dtype=float),
        chlorine_true=np.asarray(result.chlorine_true, dtype=float),
        mae_per_location=np.asarray(result.mae_per_location, dtype=float),
        node_ids=np.asarray(result.node_ids, dtype=str),
        link_ids=np.asarray(result.link_ids, dtype=str)
    )

    metrics_file = result.result_file.with_name(f"{result.dataset.file_prefix}_surrogate_metrics.json")

    metrics = {
        "dataset": result.dataset.net_name,
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
        },
    }

    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return result.result_file


def load_surrogate_evaluation_result(
    input_file: Path,
    dataset: Dataset,
) -> SurrogateEvaluationResult:
    """Load a saved surrogate evaluation result."""
    input_file = Path(input_file)

    with np.load(input_file, allow_pickle=False) as saved:
        return SurrogateEvaluationResult(
            dataset=dataset,
            chlorine_predictions=np.asarray(saved["chlorine_predictions"], dtype=float),
            chlorine_true=np.asarray(saved["chlorine_true"], dtype=float),
            mae_per_location=np.asarray(saved["mae_per_location"], dtype=float),
            node_ids=tuple(str(value) for value in saved["node_ids"].tolist()),
            link_ids=tuple(str(value) for value in saved["link_ids"].tolist()),
            result_file=input_file,
        )


class SurrogateEvaluator:
    """Evaluate trained neural surrogate model (without Kalman filter)."""

    def __init__(self, surrogate_manager: SurrogateManager):
        self.surrogates = surrogate_manager

    def run_neural_surrogate_evaluation(self, dataset: Dataset) -> SurrogateEvaluationResult:
        """Run single step prediction over test trajectory."""
        scada_file = self.surrogates.test_scada_file(dataset)
        control_file = self.surrogates.test_actions_file(dataset)

        for path in (scada_file, control_file):
            if not path.exists():
                raise FileNotFoundError(path)

        scada = ScadaData.load_from_file(str(scada_file))

        flows = np.asarray(scada.get_data_flows(), dtype=float)
        node_quality = np.asarray(scada.get_data_nodes_quality(), dtype=float)
        link_quality = np.asarray(scada.get_data_links_quality(), dtype=float)

        with np.load(control_file, allow_pickle=False) as saved:
            control_actions = np.asarray(saved["control_actions"], dtype=float)

        n_time_steps = flows.shape[0]
        current_node_quality = node_quality[:n_time_steps - 1]
        current_link_quality = link_quality[:n_time_steps - 1]
        next_flow = flows[1:n_time_steps]

        true_chlorine_values = np.concatenate((
            node_quality[1:n_time_steps],
            link_quality[1:n_time_steps]),
            axis=1,
        )

        current_state = np.concatenate((
            current_node_quality,
            current_link_quality,
            next_flow),
            axis=1,
        )

        controls = control_actions[:n_time_steps - 1]
        baseline_module = load_baseline_module(self.surrogates.paths,"fit_surrogates")
        model = baseline_module.get_mlp_state_transition_model(dataset.net_name)
        model.load_from_file(str(self.surrogates.surrogate_file(dataset)))

        chlorine_predictions = np.asarray(
            model.predict(current_state, controls, invert_output_scaling=True), dtype=float
        )

        node_ids = tuple(str(node_id) for node_id in (scada.network_topo.get_all_nodes()))
        link_ids = tuple(str(link_id) for link_id, _ in (scada.network_topo.get_all_links()))

        mae_per_location = calculate_mae_per_location(
            chlorine_predictions=chlorine_predictions,
            true_chlorine_values=true_chlorine_values
        )

        return SurrogateEvaluationResult(
            dataset=dataset,
            chlorine_predictions=chlorine_predictions,
            chlorine_true=true_chlorine_values,
            mae_per_location=mae_per_location,
            node_ids=node_ids,
            link_ids=link_ids,
            result_file=surrogate_evaluation_file(self.surrogates.paths, dataset)
        )
