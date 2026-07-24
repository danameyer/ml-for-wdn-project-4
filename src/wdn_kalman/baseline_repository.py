"""Access to the external baseline repository."""

import importlib
import sys
from pathlib import Path
from typing import Any

from wdn_kalman.paths import ProjectPaths


def add_baseline_repository(
    paths: ProjectPaths,
) -> Path:
    """Make the external baseline modules importable."""
    repo_dir = paths.baseline_repo_dir

    required_file = (
        repo_dir
        / "run_exp_state_estimation.py"
    )

    if not required_file.is_file():
        raise FileNotFoundError(
            "The baseline repository is missing or "
            "the submodule is not initialized:\n"
            f"{repo_dir}"
        )

    repo_path = str(repo_dir)

    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    return repo_dir


def load_baseline_module(
    paths: ProjectPaths,
    module_name: str,
) -> Any:
    """Load a module from the external baseline repository."""
    add_baseline_repository(paths)
    return importlib.import_module(module_name)