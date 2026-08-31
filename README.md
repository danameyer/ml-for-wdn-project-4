# ml-for-wdn-project-4

## Setup

Clone the repository including submodules:

```bash
git clone --recurse-submodules <repository-url>
cd ml-for-wdn-project-4
```

Create and activate the virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

## External implementation

This project uses
`andreArtelt/NeuralSurrogateKalmanChlorineEstimation`
as a Git submodule under `external/`.

The submodule contains the implementation for the article:

André Artelt, Janine Strotherm, Luca Hermes, and Barbara Hammer,
“Neural Surrogate Model in an Extended Kalman Filter for Chlorine
Concentration State Estimation in Water Distribution Systems,”
SysTol 2025.

## Experiments
Training and experiment scripts are located in `\scripts` and used notebooks in `\notebooks`. Slurm jobs are located in `\slurm`