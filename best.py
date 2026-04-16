# -*- coding: utf-8 -*-
"""best.py

A Python implementation for estimating the
b-value of the Gutenberg–Richter frequency-magnitude distribution using several
alternative maximum-likelihood methods.

Implemented algorithms include Aki (1965), Kijko & Smit (2012), Taroni (2021),
van der Elst (2021), and Lippiello & Petrillo (2024).
--------------------------------------------------------------------------
Usage example (quick start)
--------------------------------------------------------------------------
--------------------------------------------------------------------------
Copyright (c) 2025, Giuseppe Falcone
MIT licence.
"""
from __future__ import annotations  # for Python 3.8 – 3.10 compatibility
import math
from typing import Sequence, Tuple, Union
import numpy as np
###############################################################################
# Public function
###############################################################################

def best(
    catalog: Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]],
    magn_column: int | None,
    method: int,
    mc: float,
    mc_column: int | None,
    delta_m: float,
    magn_bin: float,
    k: int,
) -> Tuple[float, int, float]:
    """Estimate the *b-value* and its uncertainty using various MLE formulations.

    Parameters
    ----------
    catalog
        **Either** a 1-D array-like object containing magnitudes only *or*
        a 2-D array-like structure (e.g. NumPy array, pandas DataFrame «.values»)
        in which case *magn_column* and possibly *mc_column* indicate the
        relevant columns.
    magn_column
        Zero-based index of the column holding magnitudes when *catalog* is 2-D.
        Ignored if *catalog* is 1-D (pass ``None``).
    method
        Integer selector for the algorithm:
        ``1``  – Classical MLE (Aki 1965) with Ogata-Yamashina bias correction
        ``2``  – Taroni 2021 / Kijko & Smit 2012, event-dependent MC
        ``3``  – *b-positive* (van der Elst 2021)
        ``4``  – *b-more-positive* (Lippiello & Petrillo 2024, call rejected here)
    mc
        Catalogue-wide magnitude of completeness *Mc* (methods 1, 3, 4).
    mc_column
        Column index containing per-event completeness magnitudes (method 2).
    delta_m
        **ΔM** minimum magnitude separation for methods 3 & 4 (ignored otherwise).
    magn_bin
        Bin size of the magnitude scale (e.g. 0.01 for moment Mw, 0.1 for ML).
        If you pass exactly ``0`` the code assumes an *effectively continuous*
        scale and falls back to the original Aki 1965 formula.
    k
        *K* retained for API compatibility. Not used by methods 1-3.

    Returns
    -------
    Bvalue
        Maximum-likelihood b-value.
    N
        Number of samples used in the estimation (length of *X*).
    Sigma
        One-sigma uncertainty (normal approximation, √N denominator).
    """

    # ------------------------------------------------------------------
    # 0. Normalise *catalog* into a 2-D NumPy array for uniform handling
    # ------------------------------------------------------------------
    catalog_arr = np.asarray(catalog)

    if catalog_arr.ndim == 1:
        # 1-D input → promote to shape (N, 1) so that column indexing still works
        catalog_arr = catalog_arr.reshape(-1, 1)
        magn_column = 0  # by construction
    elif catalog_arr.ndim != 2:
        raise ValueError("'catalog' must be 1-D or 2-D array-like")

    n_cols = int(catalog_arr.shape[1])

    def _validate_column_index(col: int | None, name: str) -> None:
        """Validate 0-based column indices and raise clear errors."""
        if col is None:
            raise ValueError(
                f"'{name}' cannot be None for this method. "
                f"Valid 0-based indices for this catalog are 0..{n_cols - 1}."
            )
        if not isinstance(col, (int, np.integer)):
            raise TypeError(f"'{name}' must be an integer column index, got {type(col).__name__}")
        if col < 0 or col >= n_cols:
            raise ValueError(
                f"'{name}'={col} is out of bounds for a catalog with {n_cols} columns "
                f"(valid 0-based indices: 0..{n_cols - 1}). "
                "Use 0-based indexing for Python arrays."
            )

    _validate_column_index(magn_column, "magn_column")

    # Retrieve floating-point machine epsilon (≈2.22e-16) for guard tolerance
    eps = np.finfo(float).eps

    # ------------------------------------------------------------------
    # 1. Extract the working vector *X* according to the chosen method
    # ------------------------------------------------------------------
    if method == 1:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Method 1 — Aki (1965) classical MLE with fixed completeness Mc
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # 1.1 Select magnitudes (Mw, ML, …)
        M = catalog_arr[:, magn_column]

        # 1.2 Keep only events *above* completeness (≥ Mc)
        M_compl = M[M >= (mc - eps)]

        # 1.3 X_i = M_i − Mc
        X = M_compl - mc

    elif method == 2:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Method 2 — Taroni (2021) / Kijko & Smit (2012)
        #            Event-dependent completeness (Mc varies per record)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _validate_column_index(mc_column, "mc_column")

        M = catalog_arr[:, magn_column]
        mc_vect = catalog_arr[:, mc_column]

        # Event-wise boolean mask: magnitude ≥ corresponding Mc_i
        idx = M >= (mc_vect - eps)

        X = M[idx] - mc_vect[idx]

    elif method == 3:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Method 3 — van der Elst (2021) *b-positive*
        #            Uses differences of consecutive magnitudes
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        M = catalog_arr[:, magn_column]
        M_compl = M[M >= (mc - eps)]

        # ΔM_j = M_{j+1} − M_j (vectorised via np.diff)
        M_diff = np.diff(M_compl)

        # Keep pairs whose difference meets the threshold ΔM
        M_diff_compl = M_diff[M_diff >= (delta_m - eps)]

        X = M_diff_compl - delta_m

    elif method == 4:
        raise ValueError(
            "Method 4 requires latitude/longitude coordinates and distance filtering. "
            "Use 'best_bmorepos_bootstrap(...)' instead of 'best(...)'."
        )

    else:
        raise ValueError("'method' must be 1, 2, 3, or 4")

    # ------------------------------------------------------------------
    # 2. Compute sample size *N*
    # ------------------------------------------------------------------
    N = int(X.size)  # cast to Python int for nicer printing
    if N == 0:
        raise RuntimeError("No events / pairs satisfy the completeness criteria; "
                           "cannot estimate b-value.")

    # ------------------------------------------------------------------
    # 3. Estimate the b-value (two alternative formulae)
    # ------------------------------------------------------------------
    mean_X = float(np.mean(X))  # convert to scalar Python float

    if magn_bin == 0:
        # ------------------------------------------------------------------
        # Aki 1965 with Ogata & Yamashina 1986 bias correction (continuous scale)
        # ------------------------------------------------------------------
        Bvalue = ((N - 1) / N) / (math.log(10.0) * mean_X)
    else:
        # ------------------------------------------------------------------
        # Guttorp & Hopkins 1986 formula (discrete magnitude bins)
        # ------------------------------------------------------------------
        Bvalue = 1.0 / (magn_bin * math.log(10.0)) * math.log((mean_X + magn_bin) / mean_X)

    # Optional replacement: uncomment to include *Utsu 1966* bin correction
    # Bvalue = ((N - 1) / N) / (math.log(10.0) * (mean_X + magn_bin / 2.0))

    # ------------------------------------------------------------------
    # 4. Uncertainty (σ) via normal approximation (Aki 1965)
    # ------------------------------------------------------------------
    Sigma = Bvalue / math.sqrt(N)

    # ------------------------------------------------------------------
    # 5. Return results to caller
    # ------------------------------------------------------------------
    return Bvalue, N, Sigma

"""Bootstrap b-more-positive estimator for real earthquake catalogs."""

def _validate_column_index(col: int, n_cols: int, name: str) -> None:
    """Validate a 0-based column index."""
    if not isinstance(col, (int, np.integer)):
        raise TypeError(f"'{name}' must be an integer column index, got {type(col).__name__}")
    if col < 0 or col >= n_cols:
        raise ValueError(
            f"'{name}'={col} is out of bounds for a catalog with {n_cols} columns "
            f"(valid 0-based indices: 0..{n_cols - 1})."
        )


def _great_circle_distance_km(
    lat1_deg: float,
    lon1_deg: float,
    lat2_deg: float,
    lon2_deg: float,
    earth_radius_km: float,
) -> float:
    """Great-circle distance in km using the haversine formula."""
    lat1 = math.radians(lat1_deg)
    lon1 = math.radians(lon1_deg)
    lat2 = math.radians(lat2_deg)
    lon2 = math.radians(lon2_deg)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return earth_radius_km * c


def best_bmorepos_bootstrap(
    catalog: Union[np.ndarray, Sequence[Sequence[float]]],
    magn_column: int,
    mc: float,
    delta_m: float,
    magn_bin: float,
    k: int,
    long_column: int,
    lat_column: int,
    dmax: float,
    *,
    bootstrap_num: int = 200,
    random_state: int | None = None,
    earth_radius_km: float = 6731.0,
) -> Tuple[float, int, float]:
    """Estimate b-value with b-more-positive and bootstrap uncertainty.

    Parameters use 0-based column indices.
    """
    catalog_arr = np.asarray(catalog, dtype=float)
    if catalog_arr.ndim != 2:
        raise ValueError("'catalog' must be a 2-D array-like object")

    n_cols = int(catalog_arr.shape[1])
    _validate_column_index(magn_column, n_cols, "magn_column")
    _validate_column_index(long_column, n_cols, "long_column")
    _validate_column_index(lat_column, n_cols, "lat_column")

    if k < 1:
        raise ValueError("'k' must be a positive integer")
    if bootstrap_num < 1:
        raise ValueError("'bootstrap_num' must be a positive integer")

    eps = np.finfo(float).eps

    cat_compl = catalog_arr[catalog_arr[:, magn_column] >= (mc - eps), :]
    if cat_compl.shape[0] < 2:
        raise RuntimeError("Not enough events above completeness to estimate b-value.")

    m_compl = cat_compl[:, magn_column]
    long_compl = cat_compl[:, long_column]
    lat_compl = cat_compl[:, lat_column]

    rng = np.random.default_rng(random_state)
    n_boot = np.empty(bootstrap_num, dtype=int)
    bvalue_boot = np.empty(bootstrap_num, dtype=float)

    for boot_idx in range(bootstrap_num):
        if boot_idx == 0:
            m_boot = m_compl
            long_boot = long_compl
            lat_boot = lat_compl
        else:
            ind_boot = rng.integers(0, len(m_compl), size=len(m_compl))
            m_boot = m_compl[ind_boot]
            long_boot = long_compl[ind_boot]
            lat_boot = lat_compl[ind_boot]

        x_list: list[float] = []
        n_events = len(m_boot)

        for i in range(n_events - 1):
            upper_j = min(i + k, n_events - 1)
            for j in range(i + 1, upper_j + 1):
                delta_mag = m_boot[j] - m_boot[i]
                if delta_mag < (delta_m - eps):
                    continue

                dist_km = _great_circle_distance_km(
                    lat_boot[j],
                    long_boot[j],
                    lat_boot[i],
                    long_boot[i],
                    earth_radius_km,
                )
                if dist_km <= dmax:
                    x_list.append(delta_mag - delta_m)
                    break

        x = np.asarray(x_list, dtype=float)
        n_boot[boot_idx] = int(x.size)

        if n_boot[boot_idx] == 0:
            raise RuntimeError(
                f"No valid event pairs found in bootstrap iteration {boot_idx + 1}."
            )

        mean_x = float(np.mean(x))

        if magn_bin == 0:
            bvalue_boot[boot_idx] = (
                ((n_boot[boot_idx] - 1) / n_boot[boot_idx])
                / (math.log(10.0) * mean_x)
            )
        else:
            bvalue_boot[boot_idx] = (
                1.0
                / (magn_bin * math.log(10.0))
                * math.log((mean_x + magn_bin) / mean_x)
            )

    n_used = int(n_boot[0])
    bvalue = float(bvalue_boot[0])
    sigma = float(np.std(bvalue_boot, ddof=1))

    return bvalue, n_used, sigma


###############################################################################
# Convenient *alias* for backward compatibility (optional)
###############################################################################

# Uppercase alias for compatibility.
BEST = best
BEST_bmorepos_bootstrap = best_bmorepos_bootstrap

###############################################################################
# Script mode: self-test with a tiny synthetic catalogue (runs when executed
# *directly*, but not when imported). This doubles as a living example.
###############################################################################

__all__: Sequence[str] = ("magnitudes_simulation",)

###############################################################################
# Public API
###############################################################################

def magnitudes_simulation(
    num_ev: int,
    b_value: float,
    magn_min: float,
    magn_bin: float = 0.0,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate earthquake magnitudes above a completeness threshold.
    Parameters
    ----------
    num_ev
        Number of events to simulate for the catalogue.
    b_value
        *b*-value of the Gutenberg–Richter law.
    magn_min
        Minimum magnitude (i.e. magnitude of completeness *Mc*). All output
        magnitudes will be ≥ this value (after binning/rounding).
    magn_bin
        Size of the magnitude bin. Allowed values:

        - ``0.0`` ⇒ treat magnitudes as continuous (no rounding),
        - ``0.1`` ⇒ round to the nearest 0.1,
        - ``0.01`` ⇒ round to the nearest 0.01.

        Any other value raises ``ValueError``.
    rng
        Optional ``numpy.random.Generator`` instance. Providing a generator
        allows *deterministic* simulations (by seeding it externally) and
        avoids interference with the global random state. When omitted the
        default NumPy RNG is used.

    Returns
    -------
    ``np.ndarray``
        1-D array of length ``num_ev`` containing the simulated magnitudes.

    Notes
    -----
    The cumulative distribution function (CDF) of magnitudes *M* that follow
    the Gutenberg–Richter law with *b*-value *b* and completeness magnitude
    *Mc* can be written as::

        P(M ≥ x) = 10^{-b (x − Mc)},   for x ≥ Mc.

    Using the inverse-transform method we draw *U ∼ Uniform(0, 1)* and compute::

        M = Mc - (1/b) log10(U).

    In practice we substitute the natural logarithm with ``np.log`` and divide by
    ``math.log(10)`` to convert into base-10. The MATLAB code incorporates a
    correction term ``- Magn_Bin / 2`` that ensures the *mean* magnitude of each
    discrete bin matches that of the underlying continuous distribution after
    rounding; we preserve this adjustment verbatim.
    """

    # ------------------------------------------------------------------
    # 0. Validate inputs early (fail-fast)
    # ------------------------------------------------------------------
    if num_ev <= 0:
        raise ValueError("'num_ev' must be a positive integer")

    if b_value <= 0:
        raise ValueError("'b_value' must be positive (physically > 0)")

    if magn_bin not in (0.0, 0.1, 0.01):
        raise ValueError("'magn_bin' must be 0, 0.1, or 0.01")

    if rng is None:
        rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # 1. Draw *num_ev* uniform random variates in (0, 1)
    # ------------------------------------------------------------------
    u = rng.random(num_ev)

    # ------------------------------------------------------------------
    # 2. Inverse CDF transformation (continuous magnitudes)
    # ------------------------------------------------------------------
    # Implementation detail: MATLAB uses *log* (base-e) so we convert log10 by
    # dividing by ln(10). The closed-form is identical.
    #
    #   M = Mc - (1 / (b · log(10))) · ln(U)
    #
    # The MATLAB code subtracts *Magn_Bin/2* prior to rounding so that the
    # expected value of the rounded variable equals that of the continuous one;
    # we replicate the same trick.
    # ------------------------------------------------------------------
    M0 = (
        -1.0 / (b_value * math.log(10.0)) * np.log(1.0 - u)
        - magn_bin / 2.0
        + magn_min
    )

    # ------------------------------------------------------------------
    # 3. Apply binning / rounding as requested
    # ------------------------------------------------------------------
    if magn_bin == 0.0:
        M = M0  # continuous scale, nothing to round
    elif magn_bin == 0.1:
        M = np.round(M0 * 10.0) / 10.0
    else:  # magn_bin == 0.01
        M = np.round(M0 * 100.0) / 100.0

    return M
