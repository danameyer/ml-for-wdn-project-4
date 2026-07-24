"""Plot aggregated Kalman-filter results."""

from pathlib import Path

import matplotlib.pyplot as plt

from wdn_kalman.paths import ProjectPaths
from wdn_kalman.results import ExperimentResult


class BaselinePlot:
    """Plot baseline experiment results."""

    def __init__(
        self,
        paths: ProjectPaths,
    ):
        self.paths = paths

    def plot_file(
            self,
            result: ExperimentResult,
            kalman_type: str,
    ) -> Path:
        """Return the output path for a result plot."""
        ensemble_part = ""

        if result.ensemble_size is not None:
            ensemble_part = (
                f"ensemble_size="
                f"{result.ensemble_size}_"
            )

        return (
                self.paths.plots_dir
                / (
                    f"baseline_"
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
        show: bool = True,
    ) -> Path:
        """Plot mean error and standard deviation."""
        plot_file = self.plot_file(
            result,
            kalman_type,
        )

        plot_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure, axes = plt.subplots(
            figsize=(7, 4)
        )

        axes.errorbar(
            result.sensors,
            result.mean_scores,
            yerr=result.std_scores,
            marker="o",
            capsize=4,
        )

        axes.set_xlabel(
            "Number of sensors per type"
        )
        axes.set_ylabel(
            "Median absolute chlorine estimation error"
        )
        axes.set_title(
            f"{kalman_type} with neural surrogate: "
            f"{result.dataset.net_name}"
        )
        axes.grid(True)

        figure.tight_layout()
        figure.savefig(
            plot_file,
            dpi=200,
            bbox_inches="tight",
        )

        if show:
            plt.show()

        plt.close(figure)

        print(
            f"Saved plot to: {plot_file}"
        )

        return plot_file