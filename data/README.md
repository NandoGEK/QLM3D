# Data

The chains from which the figures in `plots/` are made, as HDF5 files: 50334 chains of 50000
sweeps each at `beta` = 2 L and `lam` = 0.25 and 0.5 (snapshot of 18 November 2026, 17:11).
`plots/convert.py` writes the files from chain files (`.jsonl`), and `plots/import_data.py` reads
them. Nothing is averaged in the conversion.

A sector is named by its start: `g` no charges and no flux, `w1_0_0` one unit of flux, `q<R>` two
static charges, -1 at the origin and +1 at (R, 0, 0).

## `chains.h5`

One group per sector, `L<L>/lam<lam>/<start>`, for example `L16/lam0.25/q8`, with the attributes
`L`, `lam`, `beta` and `start`. The first axis of every dataset runs over the chains.

| dataset | shape | content |
| --- | --- | --- |
| `seed` | (chains,) | seed of the chain |
| `n_burn` | (chains,) | thermalization sweeps before the first bin |
| `energy` | (chains, 20) | mean energy of every bin of 2500 sweeps |
| `correlator` | (chains, 20, 9) | only in `g`: the photon correlator of every bin at tau = j `beta` / 16, j = 0 ... 8 |

The analysis leaves out the first bins of every chain as thermalization: 2 bins for L < 10, 4 for
L = 10 and 8 for L >= 12.

## `fields.h5`

One group per sector with charges, `L<L>/lam<lam>/q<R>`, with the attributes `L`, `lam`, `R` and
`n_sweeps`.

| dataset | shape | content |
| --- | --- | --- |
| `seed` | (chains,) | seed of the chain |
| `up` | (chains, L, L, L, 3) | on the link from site (a, b, c) in direction mu: the number of the `n_sweeps` sweeps in which the link carried E = +1/2 |

The mean field of a chain on a link is `up / n_sweeps - 1/2`. The counts are
exact, and they compress to less than half the size of the mean as a floating point number.
