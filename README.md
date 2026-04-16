# BEST: Gutenberg-Richter b-value Toolkit for Python

Estimate Gutenberg-Richter b-values with multiple MLE-based methods, run a real-catalog example, and validate estimators with Monte Carlo simulations.

## Features

- Magnitude simulation following Gutenberg-Richter law (continuous or binned).
- b-value estimation with methods:
1. Aki (1965)
2. Kijko and Smit (2012) / Taroni (2021)
3. van der Elst (2021) b-positive
4. Lippiello and Petrillo (2024) b-more-positive (bootstrap version)
- Validation script that compares empirical CDF/PDF with normal approximations.

## Repository Layout

- `best.py`: core estimators (`best`, `best_bmorepos_bootstrap`) and `magnitudes_simulation`.
- `validation.py`: Monte Carlo validation workflow and plotting.
- `bvalue_real_example.py`: example on real earthquake catalogs.
- `CAT5_Inside_Depth20*.txt`: example input catalogs.

## Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

## Quick Start

Run the real-catalog example:

```bash
python bvalue_real_example.py
```

Run Monte Carlo validation:

```bash
python validation.py
```

Edit user parameters at the top of `validation.py` and `bvalue_real_example.py` to choose method, `Mc`, `delta_m`, `k`, and bootstrap settings.

## API Notes

Use `best(...)` for methods 1-3.

Method 4 requires coordinates and distance filtering, so use:

```python
from best import best_bmorepos_bootstrap
```

Normal CDF/PDF in `validation.py` are computed with Python standard library (`statistics.NormalDist`), so SciPy is not required.

## Requirements

Runtime dependencies are intentionally minimal:

- `numpy`
- `matplotlib`

