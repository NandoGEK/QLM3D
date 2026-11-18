# qlm3d

Stochastic series expansion (SSE) quantum Monte Carlo for the S = 1/2 U(1) quantum link model
with RK term on the periodic cubic lattice (3+1D):

    H = -J sum_p (U_p + U_p^dag) + lambda sum_p (U_p + U_p^dag)^2

with J = 1/2. The code uses the ratio `lam` = lambda / J (the second term is `lam * J * N_flip`),
so the RK point is `lam` = 1.

The sampler is the diagonal update plus the sweeping cluster of Yan et al., Phys. Rev. B 99,
165135 (2019). `src/sse.py` runs it in compiled kernels (numba) on an index of the operator
string: about 0.2, 2 and 7 ms per sweep at L = 4, 8, 12 and beta = 2 L. A cluster walk that gets
wide (more than 20 open lines) or long (more than 32 operators) is given up with probability 0.1
per further operator. The same algorithm in plain, readable Python, slot by slot and without
giving up walks, is `tests/reference_sse.py` (about 30 sweeps per second on 2x2x2 at beta = 4).
The tests compare the two decision for decision.

## Files

Six flat files in `src/` that import each other directly (run them from `src/`), and one test
file per source file in `tests/`. Every function has a one-sentence docstring and at least one
test named after it (`test_<function>`). The same holds for `plots/import_data.py`. The plot
scripts and `plots/convert.py` are plain scripts without tests.

| file | content |
| --- | --- |
| `src/lattice.py` | class `Cubic`: the periodic lattice, the conserved quantities, flippable plaquettes |
| `src/starts.py` | start states: random configurations with given charges and flux |
| `src/sse.py` | class `SSE`: one Markov chain of operator strings and its updates (numba) |
| `tests/reference_sse.py` | the same chain in plain Python, the reference of the tests |
| `src/observables.py` | order parameter of a configuration, photon correlator and layer winding of a string |
| `src/ed.py` | exact diagonalisation on tiny lattices |
| `src/run.py` | running chains, listing the chains of a results file, the check against exact diagonalisation, the command line |

```
lattice.py      starts.py and ed.py (use a Cubic)
    |
sse.py          one chain on a Cubic
    |
observables.py  measurements on a configuration or a chain
    |
run.py          uses all five
```

## Figures and data

`plots/` holds the scripts for the figures of the thesis and of the presentation, and `data/` the
chains they are made from, as HDF5 files (147 MB).

To make the figures:

```sh
uv sync                     # once: installs numpy, matplotlib, h5py and pandas
cd plots
uv run python sketches.py   # the sketches of the model
uv run python field.py      # the field of two static charges
uv run python energy.py     # flux energy, potential, electric coupling
uv run python photon.py     # photon correlator
```

Each script takes a few seconds and writes its figures to `plots/thesis/<name>.pdf` and
`plots/presentation/<name>.png`, over the copies that are in the repository. The scripts are
plain: constants at the top, then one block per figure. They get the data from
`plots/import_data.py`, which reads the HDF5 files into pandas DataFrames with one row per chain.

To make the figures from other chains, convert their `.jsonl` files to the HDF5 files first:

```sh
uv run python convert.py <folder with the .jsonl files>
```

[data/README.md](data/README.md) describes the layout of the data.

## Use

`uv` manages Python and the dependencies: numpy and numba for the simulation, pytest for the
tests, and matplotlib, h5py and pandas for the figures.

```sh
uv sync
uv run pytest                    # quick tests (about a minute), -m slow for the 4x4x4 chains
cd src
# One chain, printed as a JSON line. Append it to a file to keep it.
uv run python run.py chain --L 4 --lam 0.5 --beta 8 --start q2 --n_burn 500 --n_sweeps 2000
uv run python run.py chain --L 4 --lam 0.5 --beta 8 --start q2 --n_burn 500 --n_sweeps 2000 \
    --n_bins 20 --seed 2 --observables energy >> chains.jsonl

uv run python run.py jobs chains.jsonl     # the chains of a results file with their parameters
uv run python run.py check                 # SSE against exact diagonalisation on 2x2x2
uv run python run.py check --n_chains 64 --n_sweeps 20000 --n_burn 2000 --workers 8
uv run python run.py chain --help          # every option of a command
```

`--n_bins` (20), `--seed` (1) and `--observables` (`full`) have defaults, and the other options of
`chain` are required.

## Examples

In `src/`:

```python
import json
from run import Job, mean_and_error, read_rows, run_chain, run_chains

# One chain
row = run_chain(L=2, lam=0.5, beta=4.0, n_burn=100, n_sweeps=200, n_bins=4, seed=1, start="g")
energies = [b["energy"] for b in row["bins"]]  # one value per bin
print(energies, row["bad"])  # bad must be 0

# Three chains with different seeds
job = Job(L=2, lam=0.5, beta=4.0, n_burn=100, n_sweeps=200, n_bins=4, seed=1, start="q1")
rows = run_chains([job._replace(seed=seed) for seed in (1, 2, 3)])
mean, error = mean_and_error(rows, "energy")
print(f"E = {mean:.3f} +- {error:.3f}")

# Keep the rows in a file, one per line, and read them back
with open("chains.jsonl", "a") as file:
    for row in rows:
        file.write(json.dumps(row) + "\n")
rows = read_rows("chains.jsonl")
```

`run_chains(jobs, workers=3)` runs on three processes. In your own script, put that call under
`if __name__ == "__main__":`, because Python starts the worker processes by importing the script
again.

## A row

`run_chain` returns a dict, stored as one JSON line. Everything is measured on the state entering
slot 0.

| key | content |
| --- | --- |
| `L`, `lam`, `beta`, `seed`, `start`, `n_burn`, `n_sweeps`, `observables` | the parameters of the chain |
| `bins` | one dict per bin: `energy` (every sweep). With `observables = "full"` also `nflip`, `f0`, `f1`, `f2`, `Q2`, `Q4`, `ReQ3` (every sweep) and `Ipos` (one sample per bin). For the start `g` also `ct`, the photon correlator at tau = j beta / 16, j = 0 ... 8 (every 10th sweep) |
| `field` | for the starts `q<R>`: the mean of `2 E` per link over all measured sweeps |
| `bad` | the number of bins after which the charges, the flux sector or the string were wrong, which must be 0 |
| `M`, `n`, `seconds` | string length and operator number at the end of the chain, run time |

A results file holds one row per line (`.jsonl`), and `jobs` reads the parameters of every chain
from it, each chain once.

## Tests

`tests/golden.json` holds start states, exact-diagonalisation values, an operator string after
100 sweeps and whole rows of `run_chain` for fixed seeds. The tests named
`..._reference_values` require these values (measured numbers up to rounding, which differs
between processors), so a change that alters a chain does not pass unnoticed.

`tests/test_sse.py` checks the sampler in three ways. With giving up switched off and the same
random numbers, `sse.py` and `tests/reference_sse.py` must have the same state and string after
every sweep (2x2x2, 2x2x4 and, marked slow, 4x4x4, each without charges and flux, with flux and
with charges). After cluster
updates the index must equal one built anew from the string. And single moves are forced with
injected random numbers.

`run.py check` is the statistical test against exact diagonalisation on 2x2x2: with 64 chains of
2000 + 20000 sweeps per point the largest of 18 deviations was 2.4 sigma. Use at least 64 chains:
with 8 the error of the mean is itself uncertain, and single points then deviate by 4 to 5 of
these errors by chance (seen at start q1, `lam` = 0.5, where 3072 chains gave -1.8 sigma).
