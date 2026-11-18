"""Read the HDF5 files in data/ into pandas DataFrames with one row per chain (no plotting).

The layout of the files is described in data/README.md. A sector is named by its start: "g" no
charges and no flux, "w1_0_0" one unit of flux, "q<R>" two static charges at the distance R.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


def dropped_bins(L):
    """The number of bins at the start of a chain that are left out as thermalization."""
    return 8 if L >= 12 else 4 if L == 10 else 2


def groups(name):
    """The paths of the groups that hold the datasets of the file `name`, sorted."""
    paths: list[str] = []
    with h5py.File(DATA / name) as file:
        file.visit(paths.append)
    return sorted({path.rsplit("/", 1)[0] for path in paths if path.endswith("/seed")})


def chains() -> pd.DataFrame:
    """chains.h5: per chain L, lam, start, seed, n_burn, its mean energy and, in the sector "g", its mean photon correlator at tau = j beta / 16 (else None), both over the kept bins."""
    rows = []
    with h5py.File(DATA / "chains.h5") as file:
        for group in groups("chains.h5"):
            size, coupling, start = group.split("/")
            L, lam = int(size[1:]), float(coupling[3:])
            drop = dropped_bins(L)
            seeds = np.asarray(file[f"{group}/seed"])
            n_burn = np.asarray(file[f"{group}/n_burn"])
            energy = np.asarray(file[f"{group}/energy"])[:, drop:].mean(axis=1)
            correlator = [None] * len(seeds)
            if start == "g":
                correlator = list(np.asarray(file[f"{group}/correlator"])[:, drop:].mean(axis=1))
            for i in range(len(seeds)):
                rows.append(
                    {
                        "L": L,
                        "lam": lam,
                        "start": start,
                        "seed": int(seeds[i]),
                        "n_burn": int(n_burn[i]),
                        "energy": float(energy[i]),
                        "correlator": correlator[i],
                    }
                )
    return pd.DataFrame(rows)


def fields() -> pd.DataFrame:
    """fields.h5: per chain with static charges L, lam, R, seed and "field", its mean field E[a, b, c, mu] on the link from site (a, b, c) in direction mu."""
    rows = []
    with h5py.File(DATA / "fields.h5") as file:
        for group in groups("fields.h5"):
            size, coupling, start = group.split("/")
            seeds = np.asarray(file[f"{group}/seed"])
            # "up" counts the sweeps in which a link carried E = +1/2, the others have E = -1/2
            sweeps = np.asarray(file[group].attrs["n_sweeps"])
            field = (np.asarray(file[f"{group}/up"]) / sweeps - 0.5).astype(np.float32)
            for i in range(len(seeds)):
                rows.append(
                    {
                        "L": int(size[1:]),
                        "lam": float(coupling[3:]),
                        "R": int(start[1:]),
                        "seed": int(seeds[i]),
                        "field": field[i],
                    }
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for read in (chains, fields):
        frame = read()
        print(f"{read.__name__}: {len(frame)} rows, columns {list(frame.columns)}")
