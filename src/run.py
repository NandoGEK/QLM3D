"""Run chains, list the chains of a results file, and check the sampler against exact results.

run.py chain --L 4 --lam 0.5 --beta 8 --start q2 --n_burn 500 --n_sweeps 2000
run.py jobs chains.jsonl
run.py check
run.py <command> --help
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from ed import component, exact_thermal
from lattice import Cubic
from observables import (
    STATE_KEYS,
    layer_plaquettes,
    layer_winding_nonzero,
    photon_correlator,
    photon_weights,
    state_observables,
)
from sse import SSE
from starts import start_state

TAU_FRACTIONS = np.arange(9) / 16  # the photon correlator is measured at tau = j beta / 16
PHOTON_EVERY = 10  # ... on every 10th sweep of a bin
CHECK_BETA = 4.0  # inverse temperature of the check against exact diagonalisation


class Job(NamedTuple):
    """The parameters of one chain."""

    L: int
    lam: float  # lambda / J
    beta: float
    n_burn: int  # sweeps before the measurement
    n_sweeps: int  # sweeps that are measured
    n_bins: int
    seed: int
    start: str  # "g", "w<x>_<y>_<z>" or "q<R>"
    observables: str = "full"  # or "energy"


# ---------------------------------------------------------------- one chain


def run_chain(L, lam, beta, n_burn, n_sweeps, n_bins, seed, start, observables="full"):
    """Run one chain and return its parameters and measurements as a row."""
    started = time.time()
    lat = Cubic(L)
    sse = SSE(lat, lam, start_state(lat, start, seed), seed=seed)
    start_sector = sector(lat, sse.state)
    for _ in range(n_burn):
        sse.sweep(beta)

    sweeps_per_bin = max(1, n_sweeps // n_bins)
    bins = []
    state_sum = np.zeros(lat.n_links)
    bad = 0  # bins after which the string or the sector was wrong
    for _ in range(n_bins):
        means, bin_state_sum = measure_bin(sse, beta, sweeps_per_bin, observables, start == "g")
        bins.append(means)
        state_sum += bin_state_sum
        bad += not (sse.is_valid() and sector(lat, sse.state) == start_sector)

    row = {}
    if start.startswith("q"):
        row["field"] = (state_sum / (n_bins * sweeps_per_bin)).round(5).tolist()
    row.update(
        L=L,
        lam=lam,
        beta=beta,
        seed=seed,
        start=start,
        observables=observables,
        n_burn=n_burn,
        n_sweeps=n_sweeps,
        bins=bins,
        bad=bad,
        M=sse.M,
        n=sse.n,
        seconds=time.time() - started,
    )
    return row


def sector(lat, s):
    """What no update may change: the charge at every site and the flux."""
    return lat.gauss(s).tolist(), lat.winding(s)


def measure_bin(sse, beta, n_sweeps, observables, photon):
    """Do n_sweeps sweeps and return the means over them and the sum of the states."""
    lat = sse.lat
    full = observables == "full"
    totals: dict[str, float] = dict.fromkeys(("energy", *STATE_KEYS) if full else ("energy",), 0.0)
    state_sum = np.zeros(lat.n_links)
    weights = photon_weights(lat)
    correlator = np.zeros(len(TAU_FRACTIONS))
    n_correlator = 0
    for i in range(n_sweeps):
        sse.sweep(beta)
        state_sum += sse.state
        totals["energy"] += sse.energy(beta)
        if full:
            for key, value in state_observables(lat, sse.state).items():
                totals[key] += value
        if photon and i % PHOTON_EVERY == 0:
            correlator += photon_correlator(sse, weights, TAU_FRACTIONS)
            n_correlator += 1

    means: dict[str, Any] = {key: float(total) / n_sweeps for key, total in totals.items()}
    if full:
        means["Ipos"] = layer_winding_nonzero(sse, layer_plaquettes(lat))
    if photon:
        means["ct"] = (correlator / n_correlator).tolist()
    return means, state_sum


def run_chains(job_list, workers=1):
    """Run the chains of job_list, on several processes if workers > 1, and return their rows."""
    job_list = list(job_list)
    if workers > 1 and job_list:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(run_chain, *zip(*job_list, strict=True)))
    return [run_chain(*job) for job in job_list]


# ---------------------------------------------------------------- results files


def read_rows(path):
    """The rows of a .jsonl file."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def job_of(row):
    """The Job that produced a row."""
    observables = row.get("observables") or ("full" if "Q2" in row["bins"][0] else "energy")
    return Job(
        L=row["L"],
        lam=row["lam"],
        beta=row["beta"],
        n_burn=row["n_burn"],
        n_sweeps=row["n_sweeps"],
        n_bins=len(row["bins"]),
        seed=row["seed"],
        start=row["start"],
        observables=observables,
    )


def jobs(path):
    """The Jobs of the chains in a results file, each once, sorted."""
    return sorted({job_of(row) for row in read_rows(path)})


# ---------------------------------------------------------------- check


def check(n_chains=64, n_sweeps=20000, n_burn=2000, workers=8, lams=(-0.5, 0.0, 0.5)):
    """Compare SSE with exact diagonalisation on 2x2x2 and return the largest deviation in sigma."""
    lat = Cubic(2)
    worst = 0.0
    for start in ("g", "w1_0_0", "q1"):
        states = component(lat, start_state(lat, start, seed=1))
        for lam in lams:
            job = Job(
                L=2, lam=lam, beta=CHECK_BETA, n_burn=n_burn, n_sweeps=n_sweeps, n_bins=20, seed=1,
                start=start,
            )  # fmt: skip
            job_list = [job._replace(seed=seed) for seed in range(1, n_chains + 1)]
            rows = run_chains(job_list, workers)
            assert all(row["bad"] == 0 for row in rows)
            energy, n_flip = exact_thermal(lat, states, lam, CHECK_BETA, J=0.5)
            print(f"start {start}, lam = {lam}, beta = {CHECK_BETA} ({len(states)} states)")
            for key, exact in (("energy", energy), ("nflip", n_flip / lat.n_plaq)):
                mean, error = mean_and_error(rows, key)
                sigma = (mean - exact) / error
                worst = max(worst, abs(sigma))
                print(
                    f"  {key:6s} SSE {mean:.4f} +- {error:.4f}  ED {exact:.4f}  {sigma:+.2f} sigma"
                )
    print(f"largest deviation {worst:.2f} sigma: {'PASS' if worst < 3.5 else 'FAIL'}")
    return worst


def mean_and_error(rows, key):
    """The mean over chains of the observable `key` and its standard error."""
    per_chain = np.array([np.mean([b[key] for b in row["bins"]]) for row in rows])
    return per_chain.mean(), per_chain.std(ddof=1) / np.sqrt(len(per_chain))


# ---------------------------------------------------------------- command line


def main(args):
    """Run the command given on the command line."""
    parser = argparse.ArgumentParser(prog="run.py", description="SSE for the 3+1D U(1) QLM")
    commands = parser.add_subparsers(dest="command")

    chain = commands.add_parser("chain", help="run one chain and print its row as JSON")
    chain.add_argument("--L", type=int, required=True, help="lattice size")
    chain.add_argument("--lam", type=float, required=True, help="lambda / J")
    chain.add_argument("--beta", type=float, required=True, help="inverse temperature")
    chain.add_argument("--start", required=True, help="g, w<x>_<y>_<z> or q<R>")
    chain.add_argument("--n_burn", type=int, required=True, help="sweeps before measuring")
    chain.add_argument("--n_sweeps", type=int, required=True, help="sweeps that are measured")
    chain.add_argument("--n_bins", type=int, default=20)
    chain.add_argument("--seed", type=int, default=1)
    chain.add_argument("--observables", choices=("full", "energy"), default="full")

    listing = commands.add_parser("jobs", help="list the chains of a results file")
    listing.add_argument("file")

    checking = commands.add_parser("check", help="compare with exact diagonalisation on 2x2x2")
    checking.add_argument("--n_chains", type=int, default=64)
    checking.add_argument("--n_sweeps", type=int, default=20000)
    checking.add_argument("--n_burn", type=int, default=2000)
    checking.add_argument("--workers", type=int, default=8)

    options = vars(parser.parse_args(args))
    command = options.pop("command")
    if command == "chain":
        print(json.dumps(run_chain(**options)))
    elif command == "jobs":
        job_list = jobs(options["file"])
        for job in job_list:
            print(" ".join(f"{name}={value}" for name, value in job._asdict().items()))
        print(f"{len(job_list)} chains")
    elif command == "check":
        check(**options)
    else:
        parser.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])
