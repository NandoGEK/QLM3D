"""Convert chain files (.jsonl, one chain per line) to the HDF5 files in data/.

    uv run python convert.py <folder with the .jsonl files>

The files ss_*.jsonl hold rows as written by `run.py chain`. Nothing is averaged. Only chains at
other parameters are left out: those not at beta = 2 L and lam = 0.25 or 0.5. See data/README.md
for the layout of the two files that are written: chains.h5 and fields.h5.
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
LAMS = (0.25, 0.5)
folder = Path(sys.argv[1])

# The chains per sector (L, lam, start), each chain once, sorted by seed.
sectors = {}
seen = set()
for path in sorted(folder.glob("ss_*.jsonl")):
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # the incomplete last line of a run that was still being written
        key = (row["L"], row["lam"], row["start"], row["seed"], row.get("engine"))
        if row["beta"] == 2 * row["L"] and row["lam"] in LAMS and key not in seen:
            seen.add(key)  # a chain that two files hold is kept once
            sectors.setdefault(key[:3], []).append(row)
for rows in sectors.values():
    rows.sort(key=lambda row: row["seed"])

# chains.h5: per chain the seed, the thermalization sweeps, and per bin the energy and, in the
# sector without charges and flux, the photon correlator.
with h5py.File(DATA / "chains.h5", "w") as file:
    for (n, lam, start), rows in sorted(sectors.items()):
        group = file.create_group(f"L{n}/lam{lam:g}/{start}")
        group.attrs.update({"L": n, "lam": lam, "beta": rows[0]["beta"], "start": start})
        group["seed"] = [row["seed"] for row in rows]
        group["n_burn"] = [row["n_burn"] for row in rows]
        group["energy"] = [[b["energy"] for b in row["bins"]] for row in rows]
        if "ct" in rows[0]["bins"][0]:
            correlator = [[b["ct"] for b in row["bins"]] for row in rows]
            group.create_dataset("correlator", data=correlator, compression="gzip", shuffle=True)

# fields.h5: per chain with static charges, on every link the number of sweeps in which the link
# carried E = +1/2. The chain stores the mean of 2 E = +-1 over its sweeps, which is
# (2 up - n_sweeps) / n_sweeps.
with h5py.File(DATA / "fields.h5", "w") as file:
    for (n, lam, start), rows in sorted(sectors.items()):
        rows = [row for row in rows if row.get("field") is not None]
        if not rows:
            continue
        n_sweeps = rows[0]["n_sweeps"]
        assert all(row["n_sweeps"] == n_sweeps for row in rows), "chains of different length"
        # The link index is 3 site + mu with site = a + L (b + L c): the stored order is
        # [c, b, a, mu], and the file holds up[chain, a, b, c, mu].
        mean_2e = np.array([row["field"] for row in rows]).reshape(len(rows), n, n, n, 3)
        up = np.rint((mean_2e.transpose(0, 3, 2, 1, 4) + 1) / 2 * n_sweeps).astype(np.uint16)
        group = file.create_group(f"L{n}/lam{lam:g}/{start}")
        group.attrs.update({"L": n, "lam": lam, "R": int(start[1:]), "n_sweeps": n_sweeps})
        group["seed"] = [row["seed"] for row in rows]
        # Chunks of 16 whole chains: the chunks h5py picks are too small to compress well.
        chunks = (min(len(up), 16), *up.shape[1:])
        group.create_dataset(
            "up", data=up, chunks=chunks, compression="gzip", compression_opts=9, shuffle=True
        )

print(f"{sum(len(rows) for rows in sectors.values())} chains in {len(sectors)} sectors -> {DATA}")
