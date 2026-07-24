"""Datasets for experiment."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Dataset:
    """Datasets for baseline experiment."""
    net_name: str
    file_prefix: str
    sensors: List[int]


NET1 = Dataset(
    net_name='Net1',
    file_prefix='net1',
    sensors=list(range(2, 10)),
)

HANOI = Dataset(
    net_name='Hanoi',
    file_prefix='hanoi',
    sensors=list(range(2, 26)),
)

DATASETS = {
    "net1": NET1,
    "hanoi": HANOI,
}


def get_dataset(name: str) -> Dataset:
    """Return a dataset configuration by name."""
    try:
        return DATASETS[name.lower()]
    except KeyError as error:
        raise ValueError(
            f"Unknown dataset {name!r}. "
            f"Choose from: {', '.join(DATASETS)}"
        ) from error