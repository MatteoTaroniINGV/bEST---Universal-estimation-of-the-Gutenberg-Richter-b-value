# BEST: Gutenberg-Richter b-value Toolkit for Python and MATLAB

Estimate Gutenberg-Richter b-values with multiple MLE-based methods, run a real-catalog example, and validate estimators with Monte Carlo simulations.

## Paper with tool description
A full description of the bEST tool is provided in this paper, published in Seismological Research Letters: [Matteo Taroni, Davide Zaccagnino, Ilaria Spassiani, Giuseppe Falcone, Giuseppe Petrillo, Giovanni Vitale, Anna Figlioli; bEST—Universal Estimation of the Gutenberg–Richter b Value, with a MATLAB/Python Toolbox. Seismological Research Letters 2026 doi: https://doi.org/10.1785/0220250175](https://doi.org/10.1785/0220250175).
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
- `New Code and Data.zip` is the MATLAB version of the code

## Python Installation

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

## License

This project is licensed under the GPL 3.0 License. See the [LICENSE](LICENSE) file for details.

## Contributing

1. **Fork** the repository.
2. Create a feature branch: `git checkout -b feature/my‑idea`.
3. **Commit** your changes: `git commit -am 'Add awesome feature'`.
4. **Push** the branch: `git push origin feature/my‑idea`.
5. Open a **Pull Request** and describe your improvement.

All contributions — bug reports, feature ideas, documentation — are welcome! 🙌

## Collaborators

Matteo Taroni (1), Davide Zaccagnino (2,1), Ilaria Spassiani (1), Giuseppe Falcone (1), Giuseppe Petrillo (3), Giovanni Vitale (1), Anna Figlioli (1) 

1. Istituto Nazionale di Geofisica e Vulcanologia (INGV), Italy
2. Institute of Risk Analysis, Prediction and Management (Risks‐X), Southern University of Science and Technology (SUSTech), Shenzhen, Guangdong, China
3. Earth Observatory of Singapore, Nanyang Technological University, Singapore

## How to Cite this repo
If you use bEST in your research, please cite it as follows:

Taroni, M., D. Zaccagnino, I. Spassiani, G. Falcone, G. Petrillo, G. Vitale, and A. Figlioli (2026). bEST. GitHub. https://github.com/

BibTeX
If you are using BibTeX, you can use the following entry:  
@article{10.1785/0220250175,
    author = {Taroni, Matteo and Zaccagnino, Davide and Spassiani, Ilaria and Falcone, Giuseppe and Petrillo, Giuseppe and Vitale, Giovanni and Figlioli, Anna},
    title = {bEST—Universal Estimation of the Gutenberg–Richter b Value, with a MATLAB/Python Toolbox},
    journal = {Seismological Research Letters},
    year = {2026},
    month = {04},
    doi = {10.1785/0220250175},
    url = {https://doi.org/10.1785/0220250175},
    eprint = {https://pubs.geoscienceworld.org/ssa/srl/article-pdf/doi/10.1785/0220250175/7799450/srl-2025175.1.pdf},
}


- `numpy`
- `matplotlib`

