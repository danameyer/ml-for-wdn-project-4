"""Plot evaluation results."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from wdn_kalman.paths import ProjectPaths
from wdn_kalman.results import ExperimentResult


class BaselinePlot:
    """Plot baseline experiment results."""

    def __init__(self, paths: ProjectPaths):
        self.paths = paths

    def plot_file(self, result: ExperimentResult, kalman_type: str, model: str) -> Path:
        """Return the output path for a result plot."""
        model_name = (
            "gnn"
            if model == "gnn"
            else "surrogate"
        )

        ensemble_part = ""

        if result.ensemble_size is not None:
            ensemble_part = f"ensemble_size={result.ensemble_size}_"

        return (
                self.paths.plots_dir
                / (
                    f"{model_name}_"
                    f"{result.dataset.file_prefix}_"
                    f"{kalman_type.lower()}_"
                    f"n_iters={result.n_iters}_"
                    f"{ensemble_part}"
                    f"seed={result.seed}.png"
                )
        )

    def plot_experiment_results(
        self,
        result: ExperimentResult,
        kalman_type: str,
        model: str,
        show: bool = True,
    ) -> Path:
        """Plot mean error and standard deviation."""
        plot_file = self.plot_file(result=result, kalman_type=kalman_type, model=model)
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(figsize=(7, 4))
        finite_mask = np.isfinite(result.mean_scores) & np.isfinite(result.std_scores)
        plot_means = np.where(finite_mask, result.mean_scores, np.nan)
        plot_stds = np.where(finite_mask, result.std_scores, np.nan)

        axes.errorbar(result.sensors, plot_means, yerr=plot_stds, marker="o", capsize=4)
        axes.set_xticks(result.sensors)

        for sensor_count, is_finite in zip(result.sensors, finite_mask):
            if not is_finite:
                axes.text(
                    sensor_count,
                    0.03,
                    "",
                    transform=axes.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=8,
                )

        axes.set_xlabel("Number of sensors per type")
        axes.set_ylabel("Median absolute chlorine estimation error")
        model_label = "GNN"  if model == "gnn" else "neural surrogate"
        axes.set_title(f"{kalman_type} with {model_label}: {result.dataset.net_name}")
        axes.grid(True)
        figure.tight_layout()
        figure.savefig(plot_file, dpi=200, bbox_inches="tight",)

        if show:
            plt.show()

        plt.close(figure)

        print(f"Saved plot to: {plot_file}")

        return plot_file

class SurrogatePlot:
    """Plot evaluation results for neural surrogate model."""

    def __init__(self, paths: ProjectPaths):
        self.paths = paths

    def node_prediction_plot_file(self, result, node_id: str) -> Path:
        """Return output path for node prediction plot."""
        return (
                self.paths.plots_dir
                / (
                    f"surrogate_"
                    f"{result.dataset.file_prefix}_"
                    f"node={node_id}_prediction.png"
                )
        )

    def plot_node_prediction(
            self,
            result,
            node_id: str,
            show: bool = True,
            max_steps: int = 200
    ) -> Path:
        """Plot true and predicted chlorine concentration at same node."""
        node_id = str(node_id)
        node_index = result.node_index(node_id)

        plot_file = self.node_prediction_plot_file(result=result, node_id=node_id,)
        plot_file.parent.mkdir(parents=True, exist_ok=True)

        chlorine_true_values = result.chlorine_true[:max_steps, node_index]
        chlorine_predictions = result.chlorine_predictions[:max_steps, node_index]

        figure, axes = plt.subplots(figsize=(7, 4))
        axes.plot(chlorine_true_values, label="Ground truth")
        axes.plot(chlorine_predictions, label="Prediction")
        axes.set_xlabel("Time step")
        axes.set_ylabel("Chlorine concentration (mg/L)")
        axes.set_title(f"Neural surrogate: {result.dataset.net_name}, node {node_id}")
        axes.set_xticks(range(0, max_steps + 1, 20))
        axes.grid(True)
        axes.legend()
        figure.tight_layout()
        figure.savefig(plot_file, dpi=200, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(figure)
        print(f"Saved plot to: {plot_file}")

        return plot_file
