"""Saving and loading experiment results."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from wdn_kalman.datasets import Dataset


@dataclass
class ExperimentResult:
    dataset: Dataset
    sensors: Sequence[int]
    mean_scores: np.ndarray
    std_scores: np.ndarray
    n_iters: int
    seed: int
    result_file: Path
    ensemble_size: int | None = None


def save_experiment_result(
    result: ExperimentResult,
    kalman_type: str,
) -> Path:
    """Save an aggregated experiment result."""
    result.result_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_values = {
        "dataset": np.asarray(
            result.dataset.net_name
        ),
        "kalman_type": np.asarray(
            kalman_type
        ),
        "sensors": np.asarray(
            result.sensors,
            dtype=np.int64,
        ),
        "mean_scores": np.asarray(
            result.mean_scores,
            dtype=float,
        ),
        "std_scores": np.asarray(
            result.std_scores,
            dtype=float,
        ),
        "n_iters": np.asarray(
            result.n_iters,
            dtype=np.int64,
        ),
        "seed": np.asarray(
            result.seed,
            dtype=np.int64,
        ),
    }

    if result.ensemble_size is not None:
        saved_values["ensemble_size"] = np.asarray(
            result.ensemble_size,
            dtype=np.int64,
        )

    np.savez(
        result.result_file,
        **saved_values,
    )

    return result.result_file

def load_experiment_result(
    input_file: Path,
    dataset: Dataset,
) -> ExperimentResult:
    """Load an experiment result."""
    input_file = Path(input_file)

    with np.load(
        input_file,
        allow_pickle=False,
    ) as saved:
        saved_dataset = str(
            saved["dataset"].item()
        )

        if saved_dataset != dataset.net_name:
            raise ValueError(
                f"Result contains dataset "
                f"{saved_dataset!r}, but "
                f"{dataset.net_name!r} was requested."
            )

        ensemble_size = (
            int(saved["ensemble_size"].item())
            if "ensemble_size" in saved.files
            else None
        )

        return ExperimentResult(
            dataset=dataset,
            sensors=saved["sensors"].astype(
                int
            ).tolist(),
            mean_scores=np.asarray(
                saved["mean_scores"],
                dtype=float,
            ),
            std_scores=np.asarray(
                saved["std_scores"],
                dtype=float,
            ),
            n_iters=int(
                saved["n_iters"].item()
            ),
            seed=int(
                saved["seed"].item()
            ),
            result_file=input_file,
            ensemble_size=ensemble_size,
        )