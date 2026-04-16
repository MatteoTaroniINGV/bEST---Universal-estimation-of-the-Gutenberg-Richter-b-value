"""Estimate b-value from a real earthquake catalog."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from best import best
from best import best_bmorepos_bootstrap


def main() -> None:
    # Select input catalog.
    CATALOG_FILENAME = "CAT5_Inside_Depth20.txt"              # all events
    #CATALOG_FILENAME = "CAT5_Inside_Depth20_MwMin17.txt"      # above completeness
    #CATALOG_FILENAME = "CAT5_Inside_Depth20_MwMin17_STI.txt"    # above completeness + variable Mc

    # Estimation parameters (0-based column indices).
    magn_column = 5
    mc_column = 10
    MC = 1.7
    MAGN_BIN = 0.01
    DELTA_M = 0.3
    K = 5
    dmax = 100
    METHOD = 4
    long_column = 0
    lat_column = 1
    # Estimation method:
    # 1 -> Aki 1965
    # 2 -> Kijko and Smit 2012 / Taroni 2021
    # 3 -> b-positive (van der Elst 2021)
    # 4 -> b-more-positive (Lippiello and Petrillo 2024)



    catalog_path = Path(__file__).resolve().parent / CATALOG_FILENAME
    cat = np.loadtxt(catalog_path)

    if METHOD == 2 and cat.shape[1] <= mc_column:
        raise ValueError(
            f"METHOD=2 requires per-event completeness values in mc_column={mc_column}, "
            f"but '{CATALOG_FILENAME}' has only {cat.shape[1]} columns."
        )

    if METHOD == 4:
        bvalue, n_used, sigma = best_bmorepos_bootstrap(
            catalog=cat,
            magn_column=magn_column,
            mc=MC,
            delta_m=DELTA_M,
            magn_bin=MAGN_BIN,
            k=K,
            long_column=long_column,
            lat_column=lat_column,
            dmax=dmax,
        )
    else:
        bvalue, n_used, sigma = best(
            cat,
            magn_column,
            METHOD,
            MC,
            mc_column if METHOD == 2 else None,
            DELTA_M,
            MAGN_BIN,
            K,
        )

    print(f"Catalog: {CATALOG_FILENAME}")
    print(f"Method: {METHOD}")
    print(f"Bvalue = {bvalue:.6f}")
    print(f"N = {n_used}")
    print(f"Sigma = {sigma:.6f}")

if __name__ == "__main__":
    main()
