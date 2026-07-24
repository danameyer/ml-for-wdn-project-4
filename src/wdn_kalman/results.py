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


def save_experiment_result(
    result: ExperimentResult,
    kalman_type: str,
) -> Path:
    """Save an experiment result."""
    output_file = result.result_file
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        output_file,
        dataset=result.dataset.net_name,
        kalman_type=kalman_type,
        sensors=np.asarray(result.sensors),
        mean_scores=result.mean_scores,
        std_scores=result.std_scores,
        n_iters=np.asarray(result.n_iters, dtype=np.int64),
        seed=np.asarray(result.seed, dtype=np.int64),
    )

    print(f"Saved result to: {output_file}")
    return output_file

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
        )