"""Project path definitions."""

from dataclasses import dataclass
from pathlib import Path


def get_project_root() -> Path:
    """Return the root directory of the project."""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    """Paths used by the Kalman filter experiments."""

    root_dir: Path

    @classmethod
    def from_project_root(cls) -> "ProjectPaths":
        """Create paths relative to the current project."""
        return cls(root_dir=get_project_root())

    # -------------------------------------------------------------------------
    # External baseline repository
    # -------------------------------------------------------------------------

    @property
    def baseline_repo_dir(self) -> Path:
        """Location of the authors' repository submodule."""
        return (
            self.root_dir
            / "external"
            / "NeuralSurrogateKalmanChlorineEstimation"
        )

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    @property
    def data_zip(self) -> Path:
        """Portable ZIP containing the generated datasets."""
        return self.root_dir / "data" / "generated_data.zip"

    @property
    def data_dir(self) -> Path:
        return self.root_dir / "data" / "generated_data"

    # -------------------------------------------------------------------------
    # Trained surrogate models
    # -------------------------------------------------------------------------

    @property
    def models_dir(self) -> Path:
        """Directory containing trained surrogates and scalers."""
        return self.root_dir / "models"

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    @property
    def results_dir(self) -> Path:
        return self.root_dir / "results"

    @property
    def aggregated_results_dir(self) -> Path:
        return self.results_dir / "aggregated"

    @property
    def neural_surrogate_evaluation_dir(self) -> Path:
        return self.results_dir / "neural_surrogate_evaluation"

    @property
    def gnn_evaluation_dir(self) -> Path:
        return self.results_dir / "gnn_evaluation"

    @property
    def plots_dir(self) -> Path:
        return self.results_dir / "plots"

    def create_output_dirs(self) -> None:
        """Create project directories."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.aggregated_results_dir.mkdir(parents=True, exist_ok=True)
        self.neural_surrogate_evaluation_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True,exist_ok=True)
        self.gnn_evaluation_dir.mkdir(parents=True, exist_ok=True)

    def validate_inputs(self) -> None:
        """Check that project inputs are available."""
        required_paths = {
            "baseline repository": self.baseline_repo_dir,
            "extracted data": self.data_dir,
        }

        missing = [
            f"{name}: {path}"
            for name, path in required_paths.items()
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Required project inputs are missing:\n"
                + "\n".join(missing)
            )