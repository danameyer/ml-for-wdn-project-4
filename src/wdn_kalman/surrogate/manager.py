"""Neural surrogate training and file management."""

from pathlib import Path

from wdn_kalman.baseline_repository import (
    load_baseline_module
)
from wdn_kalman.datasets import Dataset
from wdn_kalman.paths import ProjectPaths


class SurrogateManager:
    """Train neural surrogate models."""

    def __init__(self, paths: ProjectPaths):
        self.paths = paths

    def training_scada_file(
        self,
        dataset: Dataset,
    ) -> Path:
        return (
            self.paths.data_dir
            / (
                f"{dataset.file_prefix}_"
                "randDemand=True_training."
                "epytflow_scada_data"
            )
        )

    def training_actions_file(
        self,
        dataset: Dataset,
    ) -> Path:
        return (
            self.paths.data_dir
            / (
                f"{dataset.file_prefix}_"
                "randDemand=True_training.npz"
            )
        )

    def test_scada_file(
        self,
        dataset: Dataset,
    ) -> Path:
        return (
            self.paths.data_dir
            / (
                f"{dataset.file_prefix}_"
                "randDemand=False_test."
                "epytflow_scada_data"
            )
        )

    def test_actions_file(
        self,
        dataset: Dataset,
    ) -> Path:
        return (
            self.paths.data_dir
            / (
                f"{dataset.file_prefix}_"
                "randDemand=False_test.npz"
            )
        )

    def surrogate_file(
        self,
        dataset: Dataset,
    ) -> Path:
        return (
            self.paths.models_dir
            / (
                f"{dataset.file_prefix}_"
                "randDemand=True_surrogate.pt"
            )
        )

    def surrogate_scaler_file(
        self,
        dataset: Dataset,
    ) -> Path:
        return Path(
            f"{self.surrogate_file(dataset)}.pickle"
        )

    def train(
            self,
            dataset: Dataset,
            overwrite: bool = False,
    ) -> Path:
        """Train and save surrogate model."""
        baseline_module = load_baseline_module(
            self.paths,
            "fit_surrogates",
        )
        fit_surrogate = baseline_module.fit_surrogate

        model_file = self.surrogate_file(dataset)
        scaler_file = self.surrogate_scaler_file(dataset)

        if (
                model_file.exists()
                and scaler_file.exists()
                and not overwrite
        ):
            print(
                f"Using existing {dataset.net_name} surrogate: "
                f"{model_file}"
            )
            return model_file

        self.paths.models_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        fit_surrogate(
            net_desc=dataset.net_name,
            scada_file_in=str(
                self.training_scada_file(dataset)
            ),
            control_actions_file_in=str(
                self.training_actions_file(dataset)
            ),
            file_out=str(model_file),
        )

        self.validate(dataset)

        print(f"Saved surrogate to: {model_file}")
        print(f"Saved scaler to: {scaler_file}")

        return model_file

    def validate(
        self,
        dataset: Dataset,
    ) -> None:
        """Check that the model and scaler are available."""
        required = [
            self.surrogate_file(dataset),
            self.surrogate_scaler_file(dataset),
        ]

        missing = [
            path
            for path in required
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing surrogate files:\n"
                + "\n".join(
                    str(path)
                    for path in missing
                )
            )