# spacecraft-pose-estimation
AI-powered autonomous spacecraft navigation using monocular vision and 6-DoF pose estimation, combining deep learning-based keypoint detection with geometric pose estimation for robust, accurate, and efficient space applications.

## Repository structure

```text
configs/         Experiment configurations
data/            HST dataset
docs/            Research documentation
notebooks/       Data exploration
results/         Evaluation outputs
scripts/         Training and evaluation
src/             Model implementations
tests/           Automated tests
README.md        Project overview
pyproject.toml   Python dependencies
```

This repository currently contains the project scaffold. The HST dataset,
model implementations, training scripts, and evaluation outputs will be added
separately. Each directory contains a `.gitkeep` placeholder so Git tracks it.

## Python environment

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate
python -m pip install -e .
```

Declare project dependencies in `pyproject.toml` as implementations are added.
No runtime dependencies are required by the initial scaffold.
