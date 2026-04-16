# -*- coding: utf-8 -*-
"""Monte Carlo validation for Gutenberg-Richter b-value estimators.

This script validates methods implemented in ``best.py``:
- method 1: Aki (1965)
- method 2: Kijko & Smit (2012) / Taroni (2021)
- method 3: van der Elst (2021) b-positive
- method 4: Lippiello & Petrillo (2024) b-more-positive bootstrap
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from best import best, best_bmorepos_bootstrap

# ---------------------------------------------------------------------------
# USER PARAMETERS
# Edit these values to run the validation without changing the functions below.
# ---------------------------------------------------------------------------
NUM_SIM = 1000
BVALUE_TRUE = 1.0
NUM_EV = 10000
MC = 1.5
MAGN_BIN = 0.0          # Allowed values: 0.0, 0.1, 0.01
DELTA_M = 0.1
K = 5
METHOD = 4              # 1, 2, 3, or 4
RANDOM_SEED = 42        # Set to None for non-reproducible runs
# Method-4-only parameters
BOOTSTRAP_NUM = 200
DMAX = 100


def magnitudes_simulation(
    num_ev: int,
    b_value: float,
    mc: float,
    magn_bin: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate synthetic magnitudes following Gutenberg-Richter law."""
    if num_ev <= 0:
        raise ValueError("'num_ev' must be a positive integer")
    if b_value <= 0:
        raise ValueError("'b_value' must be positive")
    if magn_bin not in (0.0, 0.1, 0.01):
        raise ValueError("'magn_bin' must be 0, 0.1, or 0.01")

    if rng is None:
        rng = np.random.default_rng()

    u = rng.random(num_ev)
    m0 = -1.0 / (b_value * math.log(10.0)) * np.log(1.0 - u) - magn_bin / 2.0 + mc

    if magn_bin == 0.0:
        return m0
    if magn_bin == 0.1:
        return np.round(m0 * 10.0) / 10.0
    return np.round(m0 * 100.0) / 100.0


def _plot_empirical_cdf(ax: Axes, data: np.ndarray) -> None:
    """Plot empirical CDF."""
    sorted_data = np.sort(data)
    cdf = np.linspace(1.0 / len(data), 1.0, len(data))
    ax.plot(sorted_data, cdf, "b", linewidth=2, label="Empirical distribution")


def run_validation(
    *,
    num_sim: int | None = None,
    bvalue_true: float | None = None,
    num_ev: int | None = None,
    mc: float | None = None,
    magn_bin: float | None = None,
    delta_m: float | None = None,
    k: int | None = None,
    method: int | None = None,
    random_seed: int | None = None,
    bootstrap_num: int | None = None,
    dmax: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Monte Carlo validation for the selected method.

    If a parameter is not provided, the value from the USER PARAMETERS
    section at the top of this file is used.
    """
    num_sim = NUM_SIM if num_sim is None else num_sim
    bvalue_true = BVALUE_TRUE if bvalue_true is None else bvalue_true
    num_ev = NUM_EV if num_ev is None else num_ev
    mc = MC if mc is None else mc
    magn_bin = MAGN_BIN if magn_bin is None else magn_bin
    delta_m = DELTA_M if delta_m is None else delta_m
    k = K if k is None else k
    method = METHOD if method is None else method
    random_seed = RANDOM_SEED if random_seed is None else random_seed
    bootstrap_num = BOOTSTRAP_NUM if bootstrap_num is None else bootstrap_num
    dmax = DMAX if dmax is None else dmax

    if method not in (1, 2, 3, 4):
        raise ValueError("'method' must be 1, 2, 3, or 4")

    b_estimates = np.empty(num_sim, dtype=float)
    n_estimates = np.empty(num_sim, dtype=float)
    sigma_estimates = np.empty(num_sim, dtype=float)

    rng = np.random.default_rng(random_seed)
    print("Running Monte Carlo validation...", file=sys.stderr)

    for i in range(num_sim):
        magnitudes = magnitudes_simulation(num_ev, bvalue_true, mc, magn_bin, rng)

        try:
            if method == 2:
                mc_vector = np.full_like(magnitudes, mc)
                catalog = np.column_stack((magnitudes, mc_vector))
                b_val, n_used, sigma_val = best(
                    catalog=catalog,
                    magn_column=0,
                    method=method,
                    mc=mc,
                    mc_column=1,
                    delta_m=delta_m,
                    magn_bin=magn_bin,
                    k=k,
                )
            elif method == 4:
                # Synthetic validation catalog:
                # magnitudes + zero longitudes + zero latitudes.
                catalog = np.column_stack((magnitudes, np.zeros_like(magnitudes), np.zeros_like(magnitudes)))
                b_val, n_used, sigma_val = best_bmorepos_bootstrap(
                    catalog=catalog,
                    magn_column=0,
                    mc=mc,
                    delta_m=delta_m,
                    magn_bin=magn_bin,
                    k=k,
                    long_column=1,
                    lat_column=2,
                    dmax=dmax,
                    bootstrap_num=bootstrap_num,
                    random_state=None if random_seed is None else random_seed + i,
                )
            else:
                b_val, n_used, sigma_val = best(
                    catalog=magnitudes,
                    magn_column=None,
                    method=method,
                    mc=mc,
                    mc_column=None,
                    delta_m=delta_m,
                    magn_bin=magn_bin,
                    k=k,
                )
        except Exception as err:
            sys.exit(f"Simulation failed at iteration {i}: {err}")

        b_estimates[i] = b_val
        n_estimates[i] = n_used
        sigma_estimates[i] = sigma_val

        if method == 4:
            print(num_sim - i - 1)
        elif (i + 1) % max(1, num_sim // 10) == 0 or i == num_sim - 1:
            print(f"Processed {i + 1}/{num_sim} catalogs", file=sys.stderr)

    mean_b_est = float(np.mean(b_estimates))
    print("\n=== Summary ===")
    print(f"True   b-value  : {bvalue_true:.6f}")
    print(f"Mean   estimate : {mean_b_est:.6f}")
    print(f"Std.   of est.  : {float(np.std(b_estimates, ddof=1)):.6f}")
    if method == 4:
        print(f"Mean bootstrap sigma: {float(np.mean(sigma_estimates)):.6f}")
    print("================\n")

    fig, (ax_cdf, ax_pdf) = plt.subplots(1, 2, figsize=(12, 5))

    _plot_empirical_cdf(ax_cdf, b_estimates)

    x_grid = np.arange(float(np.min(b_estimates)), float(np.max(b_estimates)) + 0.001, 0.001)
    sigma_aki = float(mean_b_est / math.sqrt(float(np.mean(n_estimates))))
    if sigma_aki <= 0:
        y_aki_cdf = np.zeros_like(x_grid)
        y_aki_pdf = np.zeros_like(x_grid)
    else:
        dist_aki = NormalDist(mean_b_est, sigma_aki)
        y_aki_cdf = np.fromiter(
            (dist_aki.cdf(float(xi)) for xi in x_grid),
            dtype=float,
            count=len(x_grid),
        )
        y_aki_pdf = np.fromiter(
            (dist_aki.pdf(float(xi)) for xi in x_grid),
            dtype=float,
            count=len(x_grid),
        )
    ax_cdf.plot(x_grid, y_aki_cdf, "--r", linewidth=2, label="Normal approximation")

    y_boot_cdf = None
    y_boot_pdf = None
    if method == 4:
        sigma_boot = float(np.mean(sigma_estimates))
        if sigma_boot <= 0:
            y_boot_cdf = np.zeros_like(x_grid)
            y_boot_pdf = np.zeros_like(x_grid)
        else:
            dist_boot = NormalDist(mean_b_est, sigma_boot)
            y_boot_cdf = np.fromiter(
                (dist_boot.cdf(float(xi)) for xi in x_grid),
                dtype=float,
                count=len(x_grid),
            )
            y_boot_pdf = np.fromiter(
                (dist_boot.pdf(float(xi)) for xi in x_grid),
                dtype=float,
                count=len(x_grid),
            )
        ax_cdf.plot(x_grid, y_boot_cdf, "--g", linewidth=2, label="Bootstrap method")

    ax_cdf.axvline(bvalue_true, color="k", linewidth=2, label="Real b-value")
    ax_cdf.axhline(0.5, color="0.5", linewidth=2)
    ax_cdf.set_xlim(x_grid[0], x_grid[-1])
    ax_cdf.set_title("CDF")
    ax_cdf.set_xlabel("b-value")
    ax_cdf.set_ylabel("F(b)")
    ax_cdf.legend()

    num_bins_hist = 50
    counts, bin_edges, _ = ax_pdf.hist(
        b_estimates,
        bins=num_bins_hist,
        edgecolor="k",
        facecolor="b",
        alpha=0.7,
        label="Empirical distribution",
    )
    bin_width = float(bin_edges[1] - bin_edges[0])
    coeff = bin_width * num_sim

    ax_pdf.plot(x_grid, y_aki_pdf * coeff, "--r", linewidth=2, label="Normal approximation")

    if method == 4 and y_boot_pdf is not None:
        ax_pdf.plot(x_grid, y_boot_pdf * coeff, "--g", linewidth=2, label="Bootstrap method")

    ymax = float(max(np.max(y_aki_pdf * coeff), np.max(counts)) + 5.0)
    ax_pdf.plot([bvalue_true, bvalue_true], [0.0, ymax], "k", linewidth=2, label="Real b-value")
    ax_pdf.set_ylim(0.0, ymax)
    ax_pdf.set_xlim(x_grid[0], x_grid[-1])
    ax_pdf.set_title("PDF")
    ax_pdf.set_xlabel("b-value")
    ax_pdf.set_ylabel("Count per bin")
    ax_pdf.legend()

    fig.tight_layout()

    out_dir = Path.cwd() / "figures"
    out_dir.mkdir(exist_ok=True)
    if method == 4:
        fig_path = out_dir / "bvalue_validation_bmorepos_bootstrap.png"
    else:
        fig_path = out_dir / f"bvalue_validation_method{method}.png"
    fig.savefig(fig_path, dpi=300)
    print(f"Figure saved to {fig_path}")
    plt.show()

    return b_estimates, n_estimates, sigma_estimates


if __name__ == "__main__":
    run_validation(
        num_sim=NUM_SIM,
        bvalue_true=BVALUE_TRUE,
        num_ev=NUM_EV,
        mc=MC,
        magn_bin=MAGN_BIN,
        delta_m=DELTA_M,
        k=K,
        method=METHOD,
        random_seed=RANDOM_SEED,
        bootstrap_num=BOOTSTRAP_NUM,
        dmax=DMAX,
    )
