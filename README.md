# qlm3d

Stochastic series expansion (SSE) quantum Monte Carlo for the S = 1/2 U(1) quantum link model
with RK term on the periodic cubic lattice (3+1D):

    H = -J sum_p (U_p + U_p^dag) + lambda sum_p (U_p + U_p^dag)^2

with J = 1/2. The code uses the ratio `lam` = lambda / J (the second term is `lam * J * N_flip`),
so the RK point is `lam` = 1.

This is a plain reference implementation: readable Python, no optimisation (about 30 sweeps per
second on 2x2x2 at beta = 4). The sampler is the diagonal update plus the sweeping cluster of
Yan et al., Phys. Rev. B 99, 165135 (2019).

## Files

Six flat files in `src/` that import each other directly (run them from `src/`), and one test
file per source file in `tests/`. Every function has a one-sentence docstring and at least one
test named after it (`test_<function>`).

| file | content |
| --- | --- |
| `src/lattice.py` | class `Cubic`: the periodic lattice, the conserved quantities, flippable plaquettes |
| `src/starts.py` | start states: random configurations with given charges and flux |
| `src/sse.py` | class `SSE`: one Markov chain of operator strings and its updates |
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

## Use

`uv` manages Python and the dependencies (numpy, and pytest for the tests).

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
uv run python run.py check --n_chains 8 --n_sweeps 6000 --n_burn 2000 --workers 8
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
energies = [b["energy"] for b in row["bins"]]      # one value per bin
print(energies, row["bad"])                        # bad must be 0

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
`..._reference_values` require exactly these values, so a change that alters the numbers of
a chain does not pass unnoticed. `run.py check` is the statistical test: with 8 chains of
2000 + 6000 sweeps at `lam` = 0.5 the largest deviation from exact diagonalisation was 2.0 sigma.
